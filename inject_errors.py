"""
inject_errors.py
----------------
Injects detectable text-level errors exclusively into the `raw_label_text`
field of a food-product JSONL file. nutrition_facts is never modified.

Modes:
  1 -> one error per record    -> <stem>_1error.jsonl
  2 -> two errors per record   -> <stem>_2errors.jsonl
  3 -> three errors per record -> <stem>_3errors.jsonl

Usage:
    python inject_errors.py <input.jsonl>

Error types (all applied to words inside raw_label_text):
  SWAP       – swap two adjacent letters        FLOUR   -> FLUOR
  DELETE     – delete one interior letter       WHEAT   -> WEAT
  INSERT     – insert a stray extra letter      SUGAR   -> SUGAER
  SUBSTITUTE – replace one interior letter      SALT    -> SAXT
  DOUBLE     – accidentally double a letter     PALM    -> PALLM

Each output record gains an "injected_errors" list:
  [
    {
      "error_type": "SWAP" | "DELETE" | "INSERT" | "SUBSTITUTE" | "DOUBLE",
      "field":      "raw_label_text",
      "original":   <original word>,
      "corrupted":  <corrupted word>,
      "char_index": <start index of the word in raw_label_text>
    },
    ...
  ]
"""

import copy
import json
import os
import random
import re
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MIN_WORD_LEN = 4

# Short codes / stop-words to skip even if they meet MIN_WORD_LEN
SKIP_PATTERN = re.compile(
    r"^(INS|AND|FOR|THE|ACID|ADDED|FROM|WITH|ALSO|EACH|ONLY|INTO)$"
)


# ---------------------------------------------------------------------------
# Mutation functions
# ---------------------------------------------------------------------------


def _swap(word: str, rng: random.Random) -> str | None:
    """Swap two adjacent distinct letters."""
    idxs = [
        i
        for i in range(len(word) - 1)
        if word[i].isalpha() and word[i + 1].isalpha() and word[i] != word[i + 1]
    ]
    if not idxs:
        return None
    i = rng.choice(idxs)
    lst = list(word)
    lst[i], lst[i + 1] = lst[i + 1], lst[i]
    return "".join(lst)


def _delete(word: str, rng: random.Random) -> str | None:
    """Delete one interior letter (keep first & last so word stays recognisable)."""
    idxs = [i for i in range(1, len(word) - 1) if word[i].isalpha()]
    if not idxs:
        return None
    i = rng.choice(idxs)
    return word[:i] + word[i + 1 :]


def _insert(word: str, rng: random.Random) -> str | None:
    """Insert a random letter at a random interior position."""
    pos = rng.randint(1, len(word) - 1)
    ch = rng.choice(ALPHABET)
    ch = ch if word[pos - 1].isupper() else ch.lower()
    return word[:pos] + ch + word[pos:]


def _substitute(word: str, rng: random.Random) -> str | None:
    """Replace one interior letter with a different random letter."""
    idxs = [i for i in range(1, len(word) - 1) if word[i].isalpha()]
    if not idxs:
        return None
    i = rng.choice(idxs)
    candidates = [c for c in ALPHABET if c != word[i].upper()]
    ch = rng.choice(candidates)
    ch = ch if word[i].isupper() else ch.lower()
    return word[:i] + ch + word[i + 1 :]


def _double(word: str, rng: random.Random) -> str | None:
    """Accidentally type a letter twice."""
    idxs = [
        i
        for i in range(1, len(word) - 1)
        if word[i].isalpha() and word[i] != word[i - 1]
    ]
    if not idxs:
        return None
    i = rng.choice(idxs)
    return word[:i] + word[i] + word[i:]


MUTATION_FNS = {
    "SWAP": _swap,
    "DELETE": _delete,
    "INSERT": _insert,
    "SUBSTITUTE": _substitute,
    "DOUBLE": _double,
}


# ---------------------------------------------------------------------------
# Token extractor
# ---------------------------------------------------------------------------


def extract_word_tokens(text: str) -> list:
    """Return [(start_index, word), ...] for eligible words in text."""
    tokens = []
    for m in re.finditer(r"[A-Za-z]{%d,}" % MIN_WORD_LEN, text):
        word = m.group()
        if SKIP_PATTERN.match(word.upper()):
            continue
        tokens.append((m.start(), word))
    return tokens


# ---------------------------------------------------------------------------
# Core injection
# ---------------------------------------------------------------------------


def inject_one_error(record: dict, rng: random.Random, used_words: set) -> dict | None:
    """
    Inject one error into raw_label_text of `record` (in-place).
    `used_words` tracks (start, word) pairs already mutated this record.
    Returns a metadata dict or None on failure.
    """
    text = record.get("raw_label_text", "")
    if not text:
        return None

    tokens = [(s, w) for s, w in extract_word_tokens(text) if (s, w) not in used_words]
    if not tokens:
        return None

    rng.shuffle(tokens)
    error_types = list(MUTATION_FNS.keys())
    rng.shuffle(error_types)

    for start, word in tokens:
        for error_type in error_types:
            corrupted_word = MUTATION_FNS[error_type](word, rng)
            if corrupted_word and corrupted_word != word:
                new_text = text[:start] + corrupted_word + text[start + len(word) :]
                record["raw_label_text"] = new_text
                used_words.add((start, word))
                return {
                    "error_type": error_type,
                    "field": "raw_label_text",
                    "original": word,
                    "corrupted": corrupted_word,
                    "char_index": start,
                }

    return None


def inject_errors(record: dict, n_errors: int, rng: random.Random) -> dict:
    """Deep-copy record and inject up to n_errors errors into raw_label_text."""
    rec = copy.deepcopy(record)
    rec.pop("injected_errors", None)

    used_words: set = set()
    errors_added = []

    for _ in range(n_errors):
        descriptor = inject_one_error(rec, rng, used_words)
        if descriptor is None:
            break
        errors_added.append(descriptor)

    rec["injected_errors"] = errors_added
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def process_file(input_path: str, seed: int = 42) -> None:
    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from '{input_path}'")

    stem, _ = os.path.splitext(input_path)

    for n_errors, suffix in [(1, "_1error"), (2, "_2errors"), (3, "_3errors")]:
        out_path = stem + suffix + ".jsonl"
        mode_rng = random.Random(seed + n_errors)

        with open(out_path, "w", encoding="utf-8") as out_f:
            for rec in records:
                corrupted = inject_errors(rec, n_errors, mode_rng)
                out_f.write(json.dumps(corrupted, ensure_ascii=False) + "\n")

        print(f"  Mode {n_errors}: {n_errors} error(s) per record  ->  {out_path}")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inject_errors.py <input.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    process_file(path)

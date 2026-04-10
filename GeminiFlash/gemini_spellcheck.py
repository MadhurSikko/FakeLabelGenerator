"""
gemini_spellcheck.py
---------------------
Infrastructure-only pipeline that sends the raw_label_text of every record
in a JSONL file to the Gemini API for spelling-error detection.

What this script does (pure infrastructure — no spell-check logic):
  1. Reads the input JSONL file record by record.
  2. For each record, sends raw_label_text to Gemini with MIME type
     set to application/json.
  3. Replaces the value of ground_truth_status ("PENDING_REVIEW") with
     the raw JSON object returned by Gemini.
  4. Writes the enriched records to an output JSONL file.
  5. Prints per-record and total stats on how many spelling errors
     Gemini reported it found.

Requirements:
  pip install google-genai

Usage:
  export GEMINI_API_KEY="your-key-here"
  python gemini_spellcheck.py <input.jsonl>

Output:
  <stem>_gemini_reviewed.jsonl
"""

import json
import os
import sys
import time

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = ""

MODEL = "gemini-3-flash-preview"
MIME_TYPE = "application/json"

# JSON schema that Gemini must follow in its response.
# All detection logic lives entirely in the prompt — the schema just
# enforces the output structure so we can parse it deterministically.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "spelling_errors_found": {
            "type": "integer",
            "description": "Total number of spelling errors detected in the text.",
        },
        "errors": {
            "type": "array",
            "description": "List of detected spelling errors.",
            "items": {
                "type": "object",
                "properties": {
                    "original": {
                        "type": "string",
                        "description": "The misspelled token as it appears in the text.",
                    },
                    "correction": {
                        "type": "string",
                        "description": "The correct spelling.",
                    },
                    "char_index": {
                        "type": "integer",
                        "description": "Character index in the text where the misspelled token starts.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Short surrounding text snippet for context.",
                    },
                },
                "required": ["original", "correction", "char_index", "context"],
            },
        },
        "reviewed_text": {
            "type": "string",
            "description": "The full text with all spelling errors corrected.",
        },
        "notes": {
            "type": "string",
            "description": "Any additional observations about the text quality.",
        },
    },
    "required": ["spelling_errors_found", "errors", "reviewed_text"],
}

SYSTEM_INSTRUCTION = """\
You are a food-label quality auditor specialising in Indian packaged food products.
Your only task is to detect genuine spelling errors in the ingredient text you are given.
The error can be of the following types:
    SWAP       – swap two adjacent letters        FLOUR   -> FLUOR
    DELETE     – delete one interior letter       WHEAT   -> WEAT
    INSERT     – insert a stray extra letter      SUGAR   -> SUGAER
    SUBSTITUTE – replace one interior letter      SALT    -> SAXT
    DOUBLE     – accidentally double a letter     PALM    -> PALLM

Rules:
- Report ONLY clear spelling mistakes — e.g. "VANILQLA" instead of "VANILLA",
  "WHAET" instead of "WHEAT", "SUGRA" instead of "SUGAR".
- Do NOT flag: INS codes (e.g. INS 503(ii)), percentage values, brand names,
  regional ingredient names (MAIDA, ATTA, BESAN), or legitimate abbreviations.
- Do NOT correct grammar, punctuation, capitalisation, or sentence structure.
- Respond strictly in the JSON format specified — no markdown, no extra text.
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(raw_label_text: str) -> str:
    return (
        "Examine the following food ingredient label text for spelling errors. "
        "Return your findings as a JSON object matching the required schema.\n\n"
        f"LABEL TEXT:\n{raw_label_text}"
    )


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------


def call_gemini(client: genai.Client, raw_label_text: str) -> dict:
    """
    Send raw_label_text to Gemini and return the parsed JSON response dict.
    All detection logic is handled entirely by the model — this function
    is pure infrastructure.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(raw_label_text),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type=MIME_TYPE,
            response_json_schema=RESPONSE_SCHEMA,
            temperature=0.0,  # deterministic — spell-check is not creative
        ),
    )
    response_text = response.text
    if response_text is None:
        raise ValueError("Gemini returned an empty response (response.text is None).")
    return json.loads(response_text)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def print_stats(stats: list[dict]) -> None:
    """Print per-record detection stats and a consolidated total."""
    print("\n" + "=" * 65)
    print("SPELLING ERROR DETECTION STATS")
    print("=" * 65)

    total_injected = 0
    total_detected = 0
    total_records = len(stats)
    records_with_errors = 0

    print(f"\n{'#':<5} {'Product ID':<22} {'Injected':>9} {'Detected':>9} {'Status'}")
    print("-" * 65)

    for s in stats:
        inj = s["injected_count"]
        det = s["detected_count"]
        total_injected += inj
        total_detected += det
        if inj > 0:
            records_with_errors += 1

        if inj == 0 and det == 0:
            status = "clean (no errors injected)"
        elif det == 0 and inj > 0:
            status = "MISSED ALL"
        elif det < inj:
            status = f"partial ({det}/{inj})"
        elif det == inj:
            status = "all found ✓"
        else:
            status = f"over-reported ({det} found, {inj} injected)"

        print(f"{s['index']:<5} {s['product_id']:<22} {inj:>9} {det:>9}   {status}")

    print("-" * 65)
    print(f"{'TOTAL':<28} {total_injected:>9} {total_detected:>9}")
    print()
    print(f"Records processed              : {total_records}")
    print(f"Records with injected errors   : {records_with_errors}")
    print(f"Total injected spelling errors : {total_injected}")
    print(f"Total detected by Gemini       : {total_detected}")

    if total_injected > 0:
        recall = total_detected / total_injected * 100
        print(f"Detection recall               : {recall:.1f}%")

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_file(input_path: str) -> None:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print(
            "ERROR: Paste your Gemini API key into GEMINI_API_KEY at the top of this file.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = genai.Client(api_key=GEMINI_API_KEY)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from '{input_path}'")
    print(f"Model  : {MODEL}")
    print(f"MIME   : {MIME_TYPE}\n")

    stem, _ = os.path.splitext(input_path)
    out_path = stem + "_gemini_reviewed.jsonl"

    stats = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, record in enumerate(records, start=1):
            product_id = record.get("product_id", f"record_{i}")
            raw_label_text = record.get("raw_label_text", "")
            injected_errors = record.get("injected_errors", [])
            injected_count = len(injected_errors)

            print(
                f"[{i:>3}/{len(records)}] {product_id} — calling Gemini...",
                end=" ",
                flush=True,
            )

            try:
                gemini_result = call_gemini(client, raw_label_text)
                detected_count = gemini_result.get("spelling_errors_found", 0)

                # Replace PENDING_REVIEW with the Gemini JSON response
                record["ground_truth_status"] = gemini_result

                print(f"detected={detected_count}  injected={injected_count}")

            except Exception as exc:
                print(f"ERROR: {exc}")
                # On error keep the original status and record the failure
                record["ground_truth_status"] = {
                    "error": str(exc),
                    "spelling_errors_found": 0,
                    "errors": [],
                    "reviewed_text": raw_label_text,
                }
                detected_count = 0

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats.append(
                {
                    "index": i,
                    "product_id": product_id,
                    "injected_count": injected_count,
                    "detected_count": detected_count,
                }
            )

            # Respect Gemini rate limits — small pause between records
            if i < len(records):
                time.sleep(0.5)

    print(f"\nOutput written to: {out_path}")
    print_stats(stats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gemini_spellcheck.py <input.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    process_file(path)

"""
inject_ins_errors.py
--------------------
Takes a JSONL file of food-product records and produces one output JSONL
where every INS code found in `raw_label_text` is replaced with a
*different* valid INS code from the same functional category.

Only `raw_label_text` is modified. All other fields are untouched.

Handles all real-world label formats observed in Indian food products:
  INS 503(ii)            -- explicit INS prefix, with suffix
  INS 472e               -- explicit INS prefix, letter suffix
  (330, 331)             -- bare numbers, comma-separated
  (322,471,472)          -- bare numbers, no spaces
  (500i)                 -- parenthesised suffix (no brackets around suffix)
  (150a&150d)            -- ampersand-separated
  (500&412)              -- ampersand, no spaces
  (330 & 331)            -- ampersand with spaces
  [ 150c, 150d ]         -- square brackets
  (452(1), 385)          -- mixed suffix styles
  E260, E102             -- E-number prefix
  (INS202)               -- INS prefix no space
  (INS 627, INS 631)     -- multiple INS-prefixed codes in one group

Two new keys are added to every record:
  "ins_code_found" : true  – at least one INS code detected & swapped
                   : false – no INS code found
  "ins_changes"    : list of change descriptors (empty when false)
    {
      "original_code":    "503(ii)",
      "replacement_code": "501",
      "original_name":    "ammonium bicarbonate",
      "replacement_name": "potassium carbonate / bicarbonate",
      "category":         "mineral salt",
      "char_index":       114
    }

Usage:
    python inject_ins_errors.py <input.jsonl>

Output:
    <stem>_ins_errors.jsonl
"""

import copy
import json
import os
import random
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# INS database — sourced from Wikipedia Codex Alimentarius list
# Format: { "code": ("name", [categories]) }
# Multi-category codes appear exactly ONCE with all categories listed.
# ---------------------------------------------------------------------------

INS_DB = {
    # ---- Colours ----
    "100": ("curcumins", ["colour"]),
    "101": ("riboflavins", ["colour"]),
    "102": ("tartrazine", ["colour"]),
    "110": ("sunset yellow FCF", ["colour"]),
    "120": ("cochineal / carmines", ["colour"]),
    "122": ("azorubine", ["colour"]),
    "124": ("ponceau 4R", ["colour"]),
    "127": ("erythrosine", ["colour"]),
    "129": ("allura red AC", ["colour"]),
    "132": ("indigo carmine", ["colour"]),
    "133": ("brilliant blue FCF", ["colour"]),
    "140": ("chlorophylls", ["colour"]),
    "141": ("chlorophylls copper complexes", ["colour"]),
    "150a": ("caramel I plain", ["colour"]),
    "150b": ("caramel II sulfite", ["colour"]),
    "150c": ("caramel III ammonia", ["colour"]),
    "150d": ("caramel IV sulfite ammonia", ["colour"]),
    "160a": ("beta-carotene", ["colour"]),
    "160c": ("paprika oleoresin", ["colour"]),
    "162": ("beet red", ["colour"]),
    "163": ("anthocyanins", ["colour"]),
    "171": ("titanium dioxide", ["colour"]),
    # ---- Preservatives ----
    "200": ("sorbic acid", ["preservative"]),
    "202": ("potassium sorbate", ["preservative"]),
    "203": ("calcium sorbate", ["preservative"]),
    "210": ("benzoic acid", ["preservative"]),
    "211": ("sodium benzoate", ["preservative"]),
    "212": ("potassium benzoate", ["preservative"]),
    "213": ("calcium benzoate", ["preservative"]),
    "220": ("sulfur dioxide", ["preservative"]),
    "221": ("sodium sulfite", ["preservative"]),
    "222": ("sodium bisulfite", ["preservative"]),
    "223": ("sodium metabisulfite", ["preservative"]),
    "224": ("potassium metabisulfite", ["preservative"]),
    "234": ("nisin", ["preservative"]),
    "235": ("natamycin", ["preservative"]),
    "249": ("potassium nitrite", ["preservative"]),
    "250": ("sodium nitrite", ["preservative"]),
    "251": ("sodium nitrate", ["preservative"]),
    "252": ("potassium nitrate", ["preservative"]),
    "280": ("propionic acid", ["preservative"]),
    "281": ("sodium propionate", ["preservative"]),
    "282": ("calcium propionate", ["preservative"]),
    "283": ("potassium propionate", ["preservative"]),
    # ---- Antioxidants ----
    "300": ("ascorbic acid", ["antioxidant"]),
    "301": ("sodium ascorbate", ["antioxidant"]),
    "302": ("calcium ascorbate", ["antioxidant"]),
    "304": ("ascorbyl palmitate", ["antioxidant"]),
    "307a": ("l-alpha-tocopherol", ["antioxidant"]),
    "307b": ("mixed tocopherols", ["antioxidant"]),
    "307c": ("dl-alpha-tocopherol", ["antioxidant"]),
    "310": ("propyl gallate", ["antioxidant"]),
    "319": ("tert-butylhydroquinone (TBHQ)", ["antioxidant"]),
    "320": ("butylated hydroxyanisole (BHA)", ["antioxidant"]),
    "321": ("butylated hydroxytoluene (BHT)", ["antioxidant"]),
    # 322 lecithins: genuinely antioxidant + emulsifier
    "322": ("lecithins", ["antioxidant", "emulsifier"]),
    # ---- Acidity regulators ----
    "260": ("acetic acid", ["acidity regulator"]),
    "261": ("potassium acetate", ["acidity regulator"]),
    "262": ("sodium acetate", ["acidity regulator"]),
    "263": ("calcium acetate", ["acidity regulator"]),
    "270": ("lactic acid", ["acidity regulator"]),
    "290": ("carbon dioxide", ["acidity regulator"]),
    "296": ("malic acid", ["acidity regulator"]),
    "297": ("fumaric acid", ["acidity regulator"]),
    "330": ("citric acid", ["acidity regulator"]),
    "331": ("sodium citrates", ["acidity regulator"]),
    "332": ("potassium citrates", ["acidity regulator"]),
    "333": ("calcium citrates", ["acidity regulator"]),
    "334": ("tartaric acid", ["acidity regulator"]),
    "338": ("phosphoric acid", ["acidity regulator"]),
    # ---- Emulsifiers ----
    "442": ("ammonium phosphatide", ["emulsifier"]),
    "471": ("mono- and diglycerides of fatty acids", ["emulsifier"]),
    "472a": ("acetic acid esters of mono- diglycerides", ["emulsifier"]),
    "472b": ("lactic acid esters of mono- diglycerides", ["emulsifier"]),
    "472c": ("citric acid esters of mono- diglycerides", ["emulsifier"]),
    "472d": ("tartaric acid esters of mono- diglycerides", ["emulsifier"]),
    "472e": ("diacetyltartaric acid esters of mono- diglycerides", ["emulsifier"]),
    "473": ("sucrose esters of fatty acids", ["emulsifier"]),
    "475": ("polyglycerol esters of fatty acids", ["emulsifier"]),
    "476": ("polyglycerol polyricinoleate", ["emulsifier"]),
    "477": ("propylene glycol esters", ["emulsifier"]),
    "481": ("sodium stearoyl lactylate", ["emulsifier"]),
    "482": ("calcium stearoyl lactylate", ["emulsifier"]),
    "491": ("sorbitan monostearate", ["emulsifier"]),
    "492": ("sorbitan tristearate", ["emulsifier"]),
    # ---- Mineral salts / Raising agents ----
    "339": ("sodium phosphates", ["mineral salt"]),
    "340": ("potassium phosphates", ["mineral salt"]),
    "341": ("calcium phosphates", ["mineral salt"]),
    "385": ("calcium disodium EDTA", ["mineral salt"]),
    "450": ("diphosphates", ["mineral salt"]),
    "451": ("triphosphates", ["mineral salt"]),
    "452": ("polyphosphates", ["mineral salt"]),
    "500": ("sodium carbonate / bicarbonate", ["mineral salt"]),
    "500(i)": ("sodium carbonate", ["mineral salt"]),
    "500(ii)": ("sodium bicarbonate", ["mineral salt"]),
    "501": ("potassium carbonate / bicarbonate", ["mineral salt"]),
    "501(i)": ("potassium carbonate", ["mineral salt"]),
    "501(ii)": ("potassium bicarbonate", ["mineral salt"]),
    "503": ("ammonium carbonate / bicarbonate", ["mineral salt"]),
    "503(i)": ("ammonium carbonate", ["mineral salt"]),
    "503(ii)": ("ammonium bicarbonate", ["mineral salt"]),
    "504": ("magnesium carbonate", ["mineral salt"]),
    "508": ("potassium chloride", ["mineral salt"]),
    "509": ("calcium chloride", ["mineral salt"]),
    # 516 calcium sulfate: mineral salt + flour treatment agent
    "516": ("calcium sulfate", ["mineral salt", "flour treatment agent"]),
    # ---- Flour treatment agents ----
    "920": ("L-cysteine", ["flour treatment agent"]),
    "925": ("chlorine", ["flour treatment agent"]),
    "926": ("chlorine dioxide", ["flour treatment agent"]),
    "928": ("benzoyl peroxide", ["flour treatment agent"]),
    "1100": ("amylases", ["flour treatment agent"]),
    "1101": ("proteases", ["flour treatment agent"]),
    "1101(i)": ("papain", ["flour treatment agent"]),
    "1101(ii)": ("bromelain", ["flour treatment agent"]),
    "1101(iii)": ("ficin", ["flour treatment agent"]),
    "1102": ("glucose oxidase", ["flour treatment agent"]),
    # ---- Thickeners / Stabilisers / Gums ----
    "400": ("alginic acid", ["thickener"]),
    "401": ("sodium alginate", ["thickener"]),
    "402": ("potassium alginate", ["thickener"]),
    "404": ("calcium alginate", ["thickener"]),
    "406": ("agar", ["thickener"]),
    "407": ("carrageenan", ["thickener"]),
    "410": ("locust bean gum", ["thickener"]),
    "412": ("guar gum", ["thickener"]),
    "413": ("tragacanth", ["thickener"]),
    "414": ("gum arabic", ["thickener"]),
    "415": ("xanthan gum", ["thickener"]),
    "436": ("polysorbate 65", ["thickener"]),
    "440": ("pectin", ["thickener"]),
    "460": ("microcrystalline cellulose", ["thickener"]),
    "460(i)": ("microcrystalline cellulose", ["thickener"]),
    "466": ("sodium carboxymethylcellulose", ["thickener"]),
    "1422": ("acetylated distarch adipate", ["thickener"]),
    # ---- Sweeteners ----
    "420": ("sorbitol", ["sweetener"]),
    "421": ("mannitol", ["sweetener"]),
    "950": ("acesulfame potassium", ["sweetener"]),
    "951": ("aspartame", ["sweetener"]),
    "954": ("saccharin", ["sweetener"]),
    "955": ("sucralose", ["sweetener"]),
    "960": ("steviol glycosides", ["sweetener"]),
    "965": ("maltitol", ["sweetener"]),
    "967": ("xylitol", ["sweetener"]),
    # ---- Flavour enhancers ----
    "620": ("glutamic acid", ["flavour enhancer"]),
    "621": ("monosodium glutamate (MSG)", ["flavour enhancer"]),
    "627": ("disodium guanylate", ["flavour enhancer"]),
    "631": ("disodium inosinate", ["flavour enhancer"]),
    "635": ("disodium ribonucleotides", ["flavour enhancer"]),
    "636": ("maltol", ["flavour enhancer"]),
    # ---- Anti-caking agents ----
    "536": ("potassium ferrocyanide", ["anti-caking agent"]),
    "551": ("silicon dioxide", ["anti-caking agent"]),
    "552": ("calcium silicate", ["anti-caking agent"]),
    "554": ("sodium aluminosilicate", ["anti-caking agent"]),
    "570": ("stearic acid", ["anti-caking agent"]),
    # ---- Sequestrants ----
    "327": ("calcium lactate", ["sequestrant"]),
    "331(i)": ("sodium dihydrogen citrate", ["sequestrant"]),
    "331(ii)": ("disodium hydrogen citrate", ["sequestrant"]),
    "331(iii)": ("trisodium citrate", ["sequestrant"]),
    "470": ("magnesium stearate", ["sequestrant"]),
}

# Build category -> [codes] lookup; multi-category codes registered under each
CATEGORY_TO_CODES = defaultdict(list)
for code, (name, categories) in INS_DB.items():
    for cat in categories:
        CATEGORY_TO_CODES[cat].append(code)

# ---------------------------------------------------------------------------
# Normalisation: map every surface form seen on labels to a canonical DB key
# Handles: letter suffixes (472e/472E), roman suffixes (500i/500ii/500(II)),
#          E-number prefix (E260), INS-no-space (INS202)
# ---------------------------------------------------------------------------


def normalise_code(raw: str) -> str:
    """
    Convert a raw code token to the canonical key used in INS_DB.
    e.g. '472E' -> '472e', '500i' -> '500(i)', 'E260' -> '260'
    Returns the normalised string (may not be in INS_DB if unknown).
    """
    s = raw.strip()

    # Strip E-number prefix
    s = re.sub(r"^[Ee](?=\d)", "", s)

    # Lowercase any trailing letter suffix that isn't inside parens
    # e.g. 472E -> 472e, 150D -> 150d
    s = re.sub(r"^(\d{3,4})([A-Za-z])$", lambda m: m.group(1) + m.group(2).lower(), s)

    # Normalise roman-suffix without parens: 500i -> 500(i), 500ii -> 500(ii)
    # Covers: 500i, 500ii, 500iii, 500(I), 500(II), 500(III)
    def _roman(m):
        num, rom = m.group(1), m.group(2).lower()
        return f"{num}({rom})"

    s = re.sub(r"^(\d{3,4})\(?([Ii]{1,3})\)?$", _roman, s)

    return s


# ---------------------------------------------------------------------------
# Token extraction: find every individual INS code token in the text
# Returns list of (start_index, end_index, raw_token, normalised_key)
# ---------------------------------------------------------------------------

# Single-code token pattern: a 3-4 digit number with an optional suffix
# suffix = letter(s)  OR  (roman)  OR  (digit)  e.g. 472e, 503(ii), 452(1)
_CODE_PAT = r"\d{3,4}(?:\([IiVv]+\)|\([0-9]\)|[a-zA-Z]+)?"

# Three detection strategies tried in order on each record:
# 1. Explicit "INS " prefix:   INS 503(ii)  /  INS472e  (with or without space)
# 2. Explicit "E" prefix:      E260  E1422
# 3. Bare number inside bracket group that is labelled with a class title keyword
#    or contains multiple codes (so it can't just be a percentage/year/quantity)

_INS_PREFIX = re.compile(r"\bINS\s*(" + _CODE_PAT + r")", re.IGNORECASE)
_E_PREFIX = re.compile(r"\bE(" + _CODE_PAT + r")\b", re.IGNORECASE)
# Bracket group: one or more bare codes separated by , ; & or spaces
_BARE_GROUP = re.compile(
    r"[\(\[]\s*(" + _CODE_PAT + r"(?:\s*[,;&]\s*" + _CODE_PAT + r")*)\s*[\)\]]"
)

# Class-title keywords that signal a bracket group contains INS codes
_CLASS_TITLES = re.compile(
    r"\b(?:colour|color|preservative|antioxidant|emulsifier|stabiliser|stabilizer|"
    r"thickener|sequestrant|acidity\s+regulator|raising\s+agent|anti.?caking|"
    r"anticaking|flavour\s+enhancer|flavor\s+enhancer|humectant|sweetener|"
    r"flour\s+treatment|firming\s+agent|bleaching\s+agent|sequesterant)\b",
    re.IGNORECASE,
)


def _extract_bare_group_codes(text: str) -> list:
    """
    Find bare number bracket groups that are preceded by a class-title keyword,
    OR that contain 2+ codes (strong signal they are additive codes, not amounts).
    Returns [(start, end, raw_token), ...] for each individual code token found.
    """
    results = []
    for m in _BARE_GROUP.finditer(text):
        group_start = m.start()
        group_content = m.group(1)

        # Check if preceded within 60 chars by a class-title keyword
        context_before = text[max(0, group_start - 60) : group_start]
        has_class_title = bool(_CLASS_TITLES.search(context_before))

        # Split the group into individual code tokens
        tokens = re.findall(_CODE_PAT, group_content)
        has_multiple = len(tokens) >= 2

        if not (has_class_title or has_multiple):
            continue  # skip — likely a percentage, year, or quantity

        # Locate each token's position in the original text
        search_from = m.start(1)
        for tok in tokens:
            pos = text.find(tok, search_from)
            if pos != -1:
                results.append((pos, pos + len(tok), tok))
                search_from = pos + len(tok)

    return results


def find_ins_tokens(text: str) -> list:
    """
    Return a deduplicated, position-sorted list of
    (start, end, raw_token, normalised_key)
    for every INS code found in text using all three strategies.
    """
    hits = {}  # start_pos -> (start, end, raw, norm)

    # Strategy 1: explicit INS prefix
    for m in _INS_PREFIX.finditer(text):
        raw = m.group(1)
        norm = normalise_code(raw)
        hits[m.start(1)] = (m.start(1), m.end(1), raw, norm)

    # Strategy 2: E-number prefix
    for m in _E_PREFIX.finditer(text):
        raw = m.group(1)
        norm = normalise_code(raw)
        if norm in INS_DB:  # only accept if it's a known code
            hits[m.start(1)] = (m.start(1), m.end(1), raw, norm)

    # Strategy 3: bare codes in labelled bracket groups
    for start, end, raw in _extract_bare_group_codes(text):
        norm = normalise_code(raw)
        if norm in INS_DB and start not in hits:
            hits[start] = (start, end, raw, norm)

    return sorted(hits.values(), key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Replacement lookup
# ---------------------------------------------------------------------------


def replacement_for(norm_key: str, rng: random.Random) -> tuple | None:
    """
    Pick a replacement code from the same functional category.
    Returns (new_code, new_name, category) or None.
    """
    if norm_key not in INS_DB:
        return None
    _, categories = INS_DB[norm_key]
    cats = list(categories)
    rng.shuffle(cats)
    for cat in cats:
        candidates = [c for c in CATEGORY_TO_CODES[cat] if c != norm_key]
        if candidates:
            new_code = rng.choice(candidates)
            new_name, _ = INS_DB[new_code]
            return new_code, new_name, cat
    return None


# ---------------------------------------------------------------------------
# Core record processing
# ---------------------------------------------------------------------------


def process_record(record: dict, rng: random.Random) -> dict:
    rec = copy.deepcopy(record)
    rec.pop("ins_code_found", None)
    rec.pop("ins_changes", None)

    text = rec.get("raw_label_text", "")
    tokens = find_ins_tokens(text)

    # Filter to only tokens we can actually replace
    actionable = [
        (s, e, raw, norm)
        for s, e, raw, norm in tokens
        if replacement_for(norm, random.Random()) is not None
    ]

    if not actionable:
        rec["ins_code_found"] = False
        rec["ins_changes"] = []
        return rec

    changes = []
    # Process right-to-left so earlier indices stay valid as text grows/shrinks
    for start, end, raw, norm in reversed(actionable):
        result = replacement_for(norm, rng)
        if result is None:
            continue
        new_code, new_name, category = result
        orig_name, _ = INS_DB[norm]

        current_text = rec["raw_label_text"]
        # Verify token is still at the expected position (offsets may have shifted)
        # Search in a small window around the expected position
        window_start = max(0, start - 5)
        actual_pos = current_text.find(raw, window_start)
        if actual_pos == -1:
            continue

        rec["raw_label_text"] = (
            current_text[:actual_pos] + new_code + current_text[actual_pos + len(raw) :]
        )

        changes.insert(
            0,
            {  # insert at front to keep left-to-right order
                "original_code": norm,
                "replacement_code": new_code,
                "original_name": orig_name,
                "replacement_name": new_name,
                "category": category,
                "char_index": actual_pos,
            },
        )

    if changes:
        rec["ins_code_found"] = True
        rec["ins_changes"] = changes
    else:
        rec["ins_code_found"] = False
        rec["ins_changes"] = []

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

    rng = random.Random(seed)
    stem, _ = os.path.splitext(input_path)
    out_path = stem + "_ins_errors.jsonl"

    found_count = 0
    not_found_count = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for rec in records:
            processed = process_record(rec, rng)
            out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")
            if processed["ins_code_found"]:
                found_count += 1
            else:
                not_found_count += 1

    print(f"  INS codes found & swapped : {found_count} records")
    print(f"  No INS codes detected     : {not_found_count} records")
    print(f"  Output written to         : {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inject_ins_errors.py <input.jsonl>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)
    process_file(path)

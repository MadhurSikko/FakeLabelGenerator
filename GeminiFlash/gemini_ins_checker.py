"""
gemini_ins_checker.py
----------------------
Infrastructure-only pipeline that sends the raw_label_text of every record
from an INS-error JSONL file to the Gemini API, asking it to verify whether
each INS-coded additive is correctly used for its declared functional class.

The error type injected by inject_ins_errors.py is a code SWAP — a valid INS
code replaced by a different valid INS code from the same functional category,
but one that doesn't match the class title on the label.
Example: RAISING AGENTS [INS 339, 500(ii)]
  → INS 339 is sodium phosphates (mineral salt), NOT a raising agent.
    The correct code for a raising agent would be INS 503(ii) ammonium bicarbonate.

What this script does (pure infrastructure — no detection logic):
  1. Reads the input JSONL file (output of inject_ins_errors.py).
  2. For each record, sends raw_label_text to Gemini 2.5 Pro with MIME
     type application/json.
  3. Replaces "PENDING_REVIEW" in ground_truth_status with the raw JSON
     object returned by Gemini.
  4. Writes enriched records to an output JSONL file.
  5. Prints a terminal overview comparing Gemini's findings against the
     ground truth stored in ins_changes.

Requirements:
  pip install google-genai

Usage:
  python gemini_ins_checker.py <input.jsonl>

Output:
  <stem>_gemini_ins_reviewed.jsonl
"""

import json
import os
import sys
import time

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration  — paste your Gemini API key here
# ---------------------------------------------------------------------------

GEMINI_API_KEY = ""

MODEL = "gemini-3-flash-preview"
MIME_TYPE = "application/json"

# ---------------------------------------------------------------------------
# Response schema Gemini must follow
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "ins_errors_found": {
            "type": "integer",
            "description": (
                "Total number of INS code mismatches detected — where an INS "
                "code is used under a functional class it does not belong to "
                "according to the Codex Alimentarius."
            ),
        },
        "flagged_codes": {
            "type": "array",
            "description": (
                "Each INS code that appears to be incorrectly assigned to its "
                "declared functional class on this label."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "ins_code": {
                        "type": "string",
                        "description": "The INS code as it appears in the label (e.g. '339', '920', '472e').",
                    },
                    "additive_name": {
                        "type": "string",
                        "description": "The actual name of the additive that INS code refers to.",
                    },
                    "declared_functional_class": {
                        "type": "string",
                        "description": (
                            "The functional class stated on the label for this code "
                            "(e.g. 'RAISING AGENTS', 'EMULSIFIER', 'FLOUR TREATMENT AGENTS')."
                        ),
                    },
                    "correct_functional_class": {
                        "type": "string",
                        "description": (
                            "The correct Codex Alimentarius functional class for this INS code."
                        ),
                    },
                    "expected_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Examples of INS codes that would be correct for the declared "
                            "functional class (e.g. if class is RAISING AGENTS, codes like "
                            "500, 500(ii), 501, 503, 503(ii) would be appropriate)."
                        ),
                    },
                    "char_index": {
                        "type": "integer",
                        "description": "Approximate character index in the label where this code appears.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Short surrounding text snippet showing how the code is used.",
                    },
                },
                "required": [
                    "ins_code",
                    "additive_name",
                    "declared_functional_class",
                    "correct_functional_class",
                    "context",
                ],
            },
        },
        "assessment": {
            "type": "string",
            "enum": ["CORRECT", "ERRORS_FOUND", "UNVERIFIABLE"],
            "description": (
                "Overall assessment: CORRECT = all INS codes match their declared "
                "functional classes, ERRORS_FOUND = at least one mismatch detected, "
                "UNVERIFIABLE = label text too garbled or contains no INS codes."
            ),
        },
        "notes": {
            "type": "string",
            "description": "Any additional observations about the INS usage in this label.",
        },
    },
    "required": ["ins_errors_found", "flagged_codes", "assessment"],
}

SYSTEM_INSTRUCTION = """\
You are a food regulatory compliance auditor with deep expertise in the Codex
Alimentarius International Numbering System (INS) for food additives and Indian
FSSAI labelling regulations.

Your sole task is to examine food ingredient label text and verify that every
INS-coded additive is correctly assigned to its declared functional class.

The error pattern you are specifically looking for:
  A valid INS code appears under a functional class it does NOT belong to.
  For example:
    - RAISING AGENTS [INS 339] — INS 339 is sodium phosphates (a mineral salt /
      acidity regulator), NOT a raising agent. This is an error.
    - FLOUR TREATMENT AGENTS [INS 920] — INS 920 is L-cysteine, which IS a flour
      treatment agent. This is correct.
    - EMULSIFIER [INS 477] — INS 477 is propylene glycol esters of fatty acids,
      which IS an emulsifier. This is correct.
    - EMULSIFIER [INS 442] — INS 442 is ammonium phosphatide, which IS an
      emulsifier. This is correct.

Rules:
- Cross-reference every INS code in the label against the Codex Alimentarius list.
- Flag a code ONLY when it genuinely does not belong to the functional class
  stated on the label.
- Do NOT flag codes that have multiple valid functional classes and the declared
  class is one of them (e.g. INS 322 lecithins is both emulsifier and antioxidant).
- Do NOT flag missing INS codes, spelling errors, or OCR artefacts — only
  functional class mismatches.
- Do NOT flag bare numbers in the label that are not clearly INS codes.
- If the label contains no INS codes at all, or is too garbled to parse,
  return assessment = "UNVERIFIABLE".
- Respond strictly in the JSON format specified — no markdown, no extra text.
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(raw_label_text: str) -> str:
    return (
        "Examine the following food ingredient label text. "
        "For every INS-coded additive present, verify that the INS code "
        "is correctly assigned to the functional class declared on the label. "
        "Flag any INS code that is used under the wrong functional class.\n\n"
        f"LABEL TEXT:\n{raw_label_text}"
    )


# ---------------------------------------------------------------------------
# Gemini call — pure infrastructure
# ---------------------------------------------------------------------------


def call_gemini(client: genai.Client, raw_label_text: str) -> dict:
    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(raw_label_text),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type=MIME_TYPE,
            response_json_schema=RESPONSE_SCHEMA,
            temperature=0.0,
        ),
    )
    response_text = response.text
    if response_text is None:
        raise ValueError("Gemini returned an empty response (response.text is None).")
    return json.loads(response_text)


# ---------------------------------------------------------------------------
# Terminal overview
# ---------------------------------------------------------------------------


def print_overview(stats: list) -> None:
    print("\n" + "=" * 76)
    print("INS CODE MISMATCH DETECTION — OVERVIEW")
    print("=" * 76)

    total_injected = 0
    total_detected = 0
    fully_caught = 0
    partially_caught = 0
    fully_missed = 0
    not_injected_records = 0

    print(f"\n{'#':<5} {'Product ID':<22} {'Injected':>9} {'Detected':>9}  {'Result'}")
    print("-" * 76)

    for s in stats:
        inj = s["injected_count"]
        det = s["detected_count"]

        if not s["was_injected"]:
            not_injected_records += 1
            result = "no INS codes injected"
        elif inj == 0:
            not_injected_records += 1
            result = "no INS codes injected"
        elif det == 0:
            fully_missed += 1
            result = "MISSED ✗"
        elif det >= inj:
            fully_caught += 1
            result = "all caught ✓"
        else:
            partially_caught += 1
            result = f"partial  ({det}/{inj})"

        if s["was_injected"] and inj > 0:
            total_injected += inj
            total_detected += det

        print(f"{s['index']:<5} {s['product_id']:<22} {inj:>9} {det:>9}  {result}")

    print("-" * 76)
    print(f"{'TOTAL':>36} {total_injected:>9} {total_detected:>9}")

    records_with_injection = sum(
        1 for s in stats if s["was_injected"] and s["injected_count"] > 0
    )

    print()
    print(f"  Records processed                 : {len(stats)}")
    print(f"  Records with injected INS errors  : {records_with_injection}")
    print(f"  Records with no INS injection     : {not_injected_records}")
    print()
    print(f"  Total injected INS mismatches     : {total_injected}")
    print(f"  Total detected by Gemini          : {total_detected}")
    print(f"  Fully caught (all errors found)   : {fully_caught}")
    print(f"  Partially caught                  : {partially_caught}")
    print(f"  Fully missed                      : {fully_missed}")

    if total_injected > 0:
        recall = total_detected / total_injected * 100
        print(f"  Detection recall                  : {recall:.1f}%")

    print("=" * 76 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_file(input_path: str) -> None:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print(
            "ERROR: Paste your Gemini API key into GEMINI_API_KEY at the "
            "top of this file.",
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
    out_path = stem + "_gemini_ins_reviewed.jsonl"

    stats = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, record in enumerate(records, start=1):
            product_id = record.get("product_id", f"record_{i}")
            raw_label = record.get("raw_label_text", "")
            was_injected = record.get("ins_code_found", False)
            ins_changes = record.get("ins_changes", [])
            injected_count = len(ins_changes)

            print(
                f"[{i:>3}/{len(records)}] {product_id} | "
                f"injected={injected_count} — calling Gemini...",
                end=" ",
                flush=True,
            )

            try:
                gemini_result = call_gemini(client, raw_label)
                detected_count = gemini_result.get("ins_errors_found", 0)
                assessment = gemini_result.get("assessment", "?")

                record["ground_truth_status"] = gemini_result

                print(f"detected={detected_count}  assessment={assessment}")

            except Exception as exc:
                print(f"ERROR: {exc}")
                record["ground_truth_status"] = {
                    "error": str(exc),
                    "ins_errors_found": 0,
                    "flagged_codes": [],
                    "assessment": "UNVERIFIABLE",
                }
                detected_count = 0

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats.append(
                {
                    "index": i,
                    "product_id": product_id,
                    "was_injected": was_injected,
                    "injected_count": injected_count,
                    "detected_count": detected_count,
                }
            )

            if i < len(records):
                time.sleep(0.1)

    print(f"\nOutput written to: {out_path}")
    print_overview(stats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gemini_ins_checker.py <input.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    process_file(path)

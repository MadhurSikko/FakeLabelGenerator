"""
openai_compound_detector.py
-----------------------------
Infrastructure-only pipeline that sends the raw_label_text of every record
from a fictitious-compound JSONL file to the OpenAI API, asking it to
identify any ingredient or additive compound that does not exist in the
official Codex Alimentarius / INS registry.

What this script does (pure infrastructure — no detection logic):
  1. Reads the input JSONL file (output of inject_fictitious_compounds.py).
  2. For each record, sends raw_label_text to the OpenAI API with structured
     JSON output enforced via response_format / json_schema.
  3. Replaces "PENDING_REVIEW" in ground_truth_status with the raw JSON
     object returned by OpenAI.
  4. Writes enriched records to an output JSONL file.
  5. Prints per-record and total stats on whether the model caught the
     injected fictitious compound, comparing against the ground truth
     stored in fictitious_compound_changes.

Requirements:
  pip install openai

Usage:
  python openai_compound_detector.py <input.jsonl>

Output:
  <stem>_gemini_compound_reviewed.jsonl   ← same suffix as the Gemini pipeline
                                            so downstream evaluation scripts
                                            work without any changes
"""

import json
import os
import sys
import time

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration  — paste your OpenAI API key here
# ---------------------------------------------------------------------------

OPENAI_API_KEY = ""
MODEL = "gpt-5.4-nano"

# ---------------------------------------------------------------------------
# Response schema the model must follow (JSON Schema for structured outputs)
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "fictitious_compounds_found": {
            "type": "integer",
            "description": (
                "Total number of ingredient compounds found that do not exist "
                "in the official Codex Alimentarius / INS registry."
            ),
        },
        "suspicious_compounds": {
            "type": "array",
            "description": "List of compounds flagged as non-existent or fictitious.",
            "items": {
                "type": "object",
                "properties": {
                    "compound_name": {
                        "type": "string",
                        "description": "The exact compound name as it appears in the label text.",
                    },
                    "ins_code": {
                        "type": "string",
                        "description": "The INS code associated with this compound in the label, if present.",
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Why this compound is considered fictitious or suspicious — "
                            "e.g. unrecognised INS number, compound name does not match "
                            "any known additive, name/number mismatch."
                        ),
                    },
                    "char_index": {
                        "type": "integer",
                        "description": "Approximate character index in the label text where the compound appears.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Short surrounding text snippet showing where the compound appears.",
                    },
                },
                "required": [
                    "compound_name",
                    "ins_code",
                    "reason",
                    "char_index",
                    "context",
                ],
                "additionalProperties": False,
            },
        },
        "assessment": {
            "type": "string",
            "enum": ["CLEAN", "SUSPICIOUS", "UNVERIFIABLE"],
            "description": (
                "Overall assessment: CLEAN = all compounds verified as real, "
                "SUSPICIOUS = at least one fictitious/unrecognised compound found, "
                "UNVERIFIABLE = label text too garbled or short to assess."
            ),
        },
        "notes": {
            "type": "string",
            "description": "Any additional observations about the ingredient list.",
        },
    },
    "required": [
        "fictitious_compounds_found",
        "suspicious_compounds",
        "assessment",
        "notes",
    ],
    "additionalProperties": False,
}

SYSTEM_INSTRUCTION = """\
You are a food regulatory compliance expert specialising in the Codex Alimentarius
International Numbering System (INS) for food additives and Indian FSSAI regulations.

Your task is to examine food ingredient label text and identify any compound,
additive, or ingredient that does NOT exist in the official Codex Alimentarius
INS registry or is not a recognised food ingredient.

Rules:
- Cross-reference every INS-coded additive against the official Codex Alimentarius list.
  Flag any INS number that is unassigned or does not exist (e.g. INS 305, INS 1428,
  INS 1109 are not real INS numbers).
- Flag compound names that structurally resemble real INS additives but are not
  actually registered — e.g. "propylated amylose succinate", "polysorbate 85",
  "potassium lactobionate" paired with an unassigned INS number.
- Do NOT flag legitimate, well-known INS additives such as INS 471, INS 322,
  INS 503(ii), INS 500(ii), INS 330, INS 412, INS 415, etc.
- Do NOT flag common natural ingredients (wheat flour, sugar, salt, palm oil, etc.)
  even if unusual.
- If the label text is too garbled or short to assess meaningfully, return
  assessment = "UNVERIFIABLE".
- Respond strictly in the JSON format specified — no markdown, no extra text.
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(raw_label_text: str) -> str:
    return (
        "Examine the following food ingredient label text. "
        "Identify any compounds, additives, or INS-coded ingredients that do not "
        "exist in the official Codex Alimentarius INS registry. "
        "Return your findings as a JSON object matching the required schema.\n\n"
        f"LABEL TEXT:\n{raw_label_text}"
    )


# ---------------------------------------------------------------------------
# OpenAI call — pure infrastructure
# ---------------------------------------------------------------------------


def call_openai(client: OpenAI, raw_label_text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": build_prompt(raw_label_text)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "compound_check_result",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
        temperature=0.0,
    )

    response_text = response.choices[0].message.content
    if response_text is None:
        raise ValueError("OpenAI returned an empty response (message.content is None).")
    return json.loads(response_text)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def print_stats(stats: list) -> None:
    print("\n" + "=" * 72)
    print("FICTITIOUS COMPOUND DETECTION STATS")
    print("=" * 72)

    total_injected = 0
    total_detected = 0
    total_records = len(stats)
    skipped_records = sum(1 for s in stats if not s["injected"])

    print(
        f"\n{'#':<5} {'Product ID':<22} {'Injected Compound':<32} "
        f"{'Detected':>9}  {'Result'}"
    )
    print("-" * 72)

    for s in stats:
        inj_name = s["injected_compound"] or "—"
        detected = s["detected_count"]

        if not s["injected"]:
            result = "not injected (skipped)"
        elif detected == 0:
            result = "MISSED ✗"
        elif detected >= 1:
            result = "CAUGHT ✓"
            total_detected += 1
        else:
            result = f"partial ({detected})"

        if s["injected"]:
            total_injected += 1

        # Truncate compound name for display
        display_name = (inj_name[:29] + "...") if len(inj_name) > 32 else inj_name

        print(
            f"{s['index']:<5} {s['product_id']:<22} {display_name:<32} "
            f"{detected:>9}  {result}"
        )

    print("-" * 72)
    print(f"{'TOTAL':<60} {total_detected:>3} / {total_injected}")
    print()
    print(f"Records processed                    : {total_records}")
    print(f"Records with injected compound       : {total_injected}")
    print(f"Records skipped (not injected)       : {skipped_records}")
    print(f"Compounds caught by OpenAI           : {total_detected}")
    print(f"Compounds missed by OpenAI           : {total_injected - total_detected}")

    if total_injected > 0:
        recall = total_detected / total_injected * 100
        print(f"Detection recall                     : {recall:.1f}%")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_file(input_path: str) -> None:
    if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        print(
            "ERROR: Paste your OpenAI API key into OPENAI_API_KEY at the "
            "top of this file.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from '{input_path}'")
    print(f"Model  : {MODEL}\n")

    stem, _ = os.path.splitext(input_path)
    # NOTE: Output suffix is intentionally "_gemini_compound_reviewed.jsonl" to
    # remain compatible with downstream evaluation scripts.
    out_path = stem + "_gemini_compound_reviewed.jsonl"

    stats = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, record in enumerate(records, start=1):
            product_id = record.get("product_id", f"record_{i}")
            raw_label_text = record.get("raw_label_text", "")
            changes = record.get("fictitious_compound_changes", {})
            was_injected = record.get("fictitious_compound_injected", False)
            injected_name = changes.get("compound_name") if was_injected else None
            injected_ins = changes.get("ins_code") if was_injected else None

            inj_display = (
                f"{injected_name} (INS {injected_ins})" if injected_name else "none"
            )
            print(
                f"[{i:>3}/{len(records)}] {product_id} | "
                f"injected: {inj_display[:40]} — calling OpenAI...",
                end=" ",
                flush=True,
            )

            try:
                openai_result = call_openai(client, raw_label_text)
                detected_count = openai_result.get("fictitious_compounds_found", 0)
                assessment = openai_result.get("assessment", "?")

                record["ground_truth_status"] = openai_result

                print(f"detected={detected_count}  assessment={assessment}")

            except Exception as exc:
                print(f"ERROR: {exc}")
                record["ground_truth_status"] = {
                    "error": str(exc),
                    "fictitious_compounds_found": 0,
                    "suspicious_compounds": [],
                    "assessment": "UNVERIFIABLE",
                    "notes": "",
                }
                detected_count = 0

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats.append(
                {
                    "index": i,
                    "product_id": product_id,
                    "injected": was_injected,
                    "injected_compound": injected_name,
                    "injected_ins": injected_ins,
                    "detected_count": detected_count,
                }
            )

            if i < len(records):
                time.sleep(0.1)

    print(f"\nOutput written to: {out_path}")
    print_stats(stats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python openai_compound_detector.py <input.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    process_file(path)

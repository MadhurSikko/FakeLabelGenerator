"""
inject_nutrition_errors.py
--------------------------
Takes a JSONL file of food-product records and produces one output JSONL
where nutrition_facts values are replaced with mathematically impossible
values — ones a model can detect using arithmetic alone, no external
knowledge required.

Three injection types, each targeting a distinct arithmetic rule:

  TYPE A — sugars > carbohydrates
    Sugars are a sub-component of carbohydrates. Sugars > carbs
    is physically impossible. Injected by setting sugars to
    carbs * multiplier (1.3 – 2.0).

  TYPE B — saturated_fat > total_fat
    Saturated fat is a sub-component of total fat. sat_fat > fat
    is physically impossible. Injected by setting sat_fat to
    fat * multiplier (1.3 – 2.0).

  TYPE C — energy vs macro mismatch
    FSSAI standard: Energy ≈ (fat × 9) + (carbs × 4) + (protein × 4)
    Atwater factors. A deviation > 20% of the calculated value is
    flagged as implausible. Injected by multiplying the declared
    energy by a factor that pushes it outside the ±20% tolerance
    (either far too high or far too low).

Records with missing/unparseable values for the relevant fields are
skipped for that injection type (other types may still apply).
Records with all-zero nutrition (water, plain salt) are skipped entirely.

Each record gets AT MOST ONE injection per type — so a record can have
up to 3 injections total. Each injection is independently detectable.

Two new keys are added to every record:
  "nutrition_errors_injected" : true | false
  "nutrition_changes"         : list of change descriptors (empty when false)
    {
      "type"           : "A" | "B" | "C",
      "rule"           : human-readable description of the rule broken,
      "field_modified" : which nutrition_facts key was changed,
      "original_value" : "25.5 g",
      "injected_value" : "91.0 g",
      "original_num"   : 25.5,
      "injected_num"   : 91.0,
      "unit"           : "g",
      "detection_hint" : what an evaluator should check to catch this
    }

Usage:
    python inject_nutrition_errors.py <input.jsonl>

Output:
    <stem>_nutrition_errors.jsonl
"""

import copy
import json
import os
import random
import re
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_num(val: str) -> float | None:
    """Extract the numeric part of a value string like '25.5 g'."""
    if not val or str(val).strip() in ("N/A", "", "0 g"):
        return None
    try:
        return float(re.sub(r"[^\d.]", "", str(val)))
    except ValueError:
        return None


def parse_unit(val: str) -> str:
    """Extract the unit from a value string like '25.5 g' → 'g'."""
    val = str(val).strip()
    m = re.search(r"[a-zA-Z%]+", val)
    return m.group(0) if m else "g"


def fmt(num: float, unit: str) -> str:
    """Format a number + unit back into a value string."""
    rounded = round(num, 4)
    # Drop trailing zeros after decimal
    if rounded == int(rounded):
        return f"{int(rounded)} {unit}"
    return f"{rounded} {unit}"


def is_all_zero(nf: dict) -> bool:
    """True if every numeric field is 0 — e.g. plain water or salt."""
    fields = ["energy", "fat", "saturated_fat", "carbohydrates", "sugars", "proteins"]
    nums = [parse_num(nf.get(f)) for f in fields]
    return all(n is not None and n == 0.0 for n in nums)


# ---------------------------------------------------------------------------
# Injection functions — one per type
# ---------------------------------------------------------------------------

def inject_type_a(nf: dict, rng: random.Random) -> dict | None:
    """
    TYPE A: sugars > carbohydrates
    Sets sugars = carbs * factor where factor ∈ [1.3, 2.0]
    Returns a change descriptor dict, or None if not applicable.
    """
    carbs_raw  = nf.get("carbohydrates")
    sugars_raw = nf.get("sugars")

    carbs  = parse_num(carbs_raw)
    sugars = parse_num(sugars_raw)

    # Need both fields to be present and non-zero carbs
    if carbs is None or sugars is None or carbs <= 0:
        return None

    # Don't inject if sugars already >= carbs (record already broken)
    if sugars >= carbs:
        return None

    unit   = parse_unit(sugars_raw)
    factor = round(rng.uniform(1.3, 2.0), 2)
    injected_num = round(carbs * factor, 1)

    nf["sugars"] = fmt(injected_num, unit)

    return {
        "type"           : "A",
        "rule"           : "Sugars cannot exceed total carbohydrates (sugars are a sub-component)",
        "field_modified" : "sugars",
        "original_value" : sugars_raw,
        "injected_value" : nf["sugars"],
        "original_num"   : sugars,
        "injected_num"   : injected_num,
        "unit"           : unit,
        "detection_hint" : f"sugars ({injected_num}{unit}) > carbohydrates ({carbs}{unit})",
    }


def inject_type_b(nf: dict, rng: random.Random) -> dict | None:
    """
    TYPE B: saturated_fat > total_fat
    Sets saturated_fat = fat * factor where factor ∈ [1.3, 2.0]
    Returns a change descriptor dict, or None if not applicable.
    """
    fat_raw    = nf.get("fat")
    sat_fat_raw = nf.get("saturated_fat")

    fat    = parse_num(fat_raw)
    sat_fat = parse_num(sat_fat_raw)

    if fat is None or sat_fat is None or fat <= 0:
        return None

    if sat_fat >= fat:
        return None

    unit   = parse_unit(sat_fat_raw)
    factor = round(rng.uniform(1.3, 2.0), 2)
    injected_num = round(fat * factor, 1)

    nf["saturated_fat"] = fmt(injected_num, unit)

    return {
        "type"           : "B",
        "rule"           : "Saturated fat cannot exceed total fat (saturated fat is a sub-component)",
        "field_modified" : "saturated_fat",
        "original_value" : sat_fat_raw,
        "injected_value" : nf["saturated_fat"],
        "original_num"   : sat_fat,
        "injected_num"   : injected_num,
        "unit"           : unit,
        "detection_hint" : f"saturated_fat ({injected_num}{unit}) > fat ({fat}{unit})",
    }


def inject_type_c(nf: dict, rng: random.Random) -> dict | None:
    """
    TYPE C: energy vs macro mismatch (>20% deviation from Atwater)
    Calculated energy = fat*9 + carbs*4 + protein*4
    Injects an energy value that is either too high (×2.0–3.5) or
    too low (×0.1–0.35) relative to the calculated value.
    Returns a change descriptor dict, or None if not applicable.
    """
    energy_raw  = nf.get("energy")
    fat_raw     = nf.get("fat")
    carbs_raw   = nf.get("carbohydrates")
    protein_raw = nf.get("proteins")

    energy  = parse_num(energy_raw)
    fat     = parse_num(fat_raw)
    carbs   = parse_num(carbs_raw)
    protein = parse_num(protein_raw)

    if any(x is None for x in [energy, fat, carbs, protein]):
        return None

    calculated = fat * 9 + carbs * 4 + protein * 4

    # Skip products where calculated energy is near zero (plain water etc.)
    if calculated < 5:
        return None

    # Skip if declared energy is already wildly off (don't stack on existing errors)
    tolerance = 0.20
    if energy > 0 and abs(energy - calculated) / calculated > 0.5:
        return None

    unit = parse_unit(energy_raw)

    # Randomly choose too-high or too-low
    if rng.random() < 0.5:
        # Too high: multiply declared energy by 2.0–3.5
        factor = round(rng.uniform(2.0, 3.5), 2)
        injected_num = round(energy * factor, 1)
        direction = "too high"
    else:
        # Too low: multiply declared energy by 0.1–0.35
        factor = round(rng.uniform(0.1, 0.35), 2)
        injected_num = round(energy * factor, 1)
        direction = "too low"

    nf["energy"] = fmt(injected_num, unit)

    deviation_pct = round(abs(injected_num - calculated) / calculated * 100, 1)

    return {
        "type"              : "C",
        "rule"              : "Energy must be consistent with macronutrients: (fat×9) + (carbs×4) + (protein×4) within ±20%",
        "field_modified"    : "energy",
        "original_value"    : energy_raw,
        "injected_value"    : nf["energy"],
        "original_num"      : energy,
        "injected_num"      : injected_num,
        "unit"              : unit,
        "calculated_energy" : round(calculated, 1),
        "deviation_pct"     : deviation_pct,
        "direction"         : direction,
        "detection_hint"    : (
            f"declared energy {injected_num}{unit} is {direction} — "
            f"Atwater calculation gives ~{round(calculated,1)}{unit} "
            f"({deviation_pct}% deviation, threshold 20%)"
        ),
    }


# ---------------------------------------------------------------------------
# Per-record processor
# ---------------------------------------------------------------------------

def process_record(record: dict, rng: random.Random) -> dict:
    rec = copy.deepcopy(record)
    rec.pop("nutrition_errors_injected", None)
    rec.pop("nutrition_changes", None)

    nf = rec.get("nutrition_facts", {})

    # Skip all-zero records (water, plain salt — nothing meaningful to break)
    if is_all_zero(nf):
        rec["nutrition_errors_injected"] = False
        rec["nutrition_changes"] = []
        return rec

    # Work on a mutable copy of nf
    nf_modified = copy.deepcopy(nf)
    changes = []

    # Randomly decide which injection types to attempt (each independently)
    # Use a fresh shuffle so not every record gets the same type first
    types = ["A", "B", "C"]
    rng.shuffle(types)

    for t in types:
        # Inject each type independently — all three can fire on same record
        if t == "A":
            change = inject_type_a(nf_modified, rng)
        elif t == "B":
            change = inject_type_b(nf_modified, rng)
        else:
            change = inject_type_c(nf_modified, rng)

        if change:
            changes.append(change)

    if changes:
        rec["nutrition_facts"] = nf_modified
        rec["nutrition_errors_injected"] = True
        rec["nutrition_changes"] = changes
    else:
        rec["nutrition_errors_injected"] = False
        rec["nutrition_changes"] = []

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
    out_path = stem + "_nutrition_errors.jsonl"

    injected_count   = 0
    skipped_count    = 0
    type_a_count     = 0
    type_b_count     = 0
    type_c_count     = 0
    total_injections = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for rec in records:
            processed = process_record(rec, rng)
            out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")

            changes = processed.get("nutrition_changes", [])
            if processed["nutrition_errors_injected"]:
                injected_count += 1
                total_injections += len(changes)
                type_a_count += sum(1 for c in changes if c["type"] == "A")
                type_b_count += sum(1 for c in changes if c["type"] == "B")
                type_c_count += sum(1 for c in changes if c["type"] == "C")
            else:
                skipped_count += 1

    print(f"\n  Records with injected errors    : {injected_count}")
    print(f"  Records skipped (incomplete)    : {skipped_count}")
    print(f"  Total injections                : {total_injections}")
    print(f"    Type A (sugars > carbs)       : {type_a_count}")
    print(f"    Type B (sat fat > total fat)  : {type_b_count}")
    print(f"    Type C (energy/macro mismatch): {type_c_count}")
    print(f"\n  Output written to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inject_nutrition_errors.py <input.jsonl>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)
    process_file(path)

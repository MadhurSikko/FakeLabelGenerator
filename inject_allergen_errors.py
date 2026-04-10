"""
inject_allergen_contradictions.py
----------------------------------
Injects allergen contradictions into food-product JSONL records by
modifying `raw_label_text` and `allergen_declarations` to create
detectable mismatches.

Two contradiction types (chosen randomly per record):

  TYPE A — "GHOST ALLERGEN"
    Adds an allergen to allergen_declarations whose triggering ingredient
    is NOT present in raw_label_text. The label says it contains X but
    the ingredient list has no evidence of X.

  TYPE B — "UNDECLARED ALLERGEN"
    Adds an allergenic ingredient into raw_label_text (spliced into the
    ingredient list) without adding it to allergen_declarations.
    The product contains X but never declares it.

Special case — empty / no declaration:
    If allergen_declarations is empty / "No allergens listed", an allergen
    is injected into the declaration only (ghost allergen), since there is
    no declaration to contradict with a present ingredient.

FSSAI allergen categories respected (per regulation 2.2.1(14)):
  Gluten   — wheat, rye, barley, oats, spelt
  Milk     — milk, dairy, whey, lactose, casein, cream, butter, cheese
  Egg      — egg, albumin, mayonnaise
  Fish     — fish, salmon, tuna, cod, anchovy
  Nut      — peanut, almond, walnut, cashew, pistachio, hazelnut
  Soy      — soy, soya, soybean, tofu, tempeh
  Crustacean — shrimp, prawn, crab, lobster, crayfish
  Sulphite — sulphite, sulfite, sodium metabisulphite (INS 223), potassium metabisulphite (INS 224)

FSSAI exemptions honoured — the following are NOT flagged as allergens
even if the base allergen is present:
  - Wheat-based glucose syrups / dextrose
  - Wheat-based maltodextrins
  - Glucose syrups based on barley
  - Oils and distilled alcoholic beverages derived from allergen ingredients
  - Distilled alcoholic beverages

Output adds two keys:
  "allergen_contradiction_injected": true | false
  "allergen_contradiction_changes": {
      "type":                "GHOST_ALLERGEN" | "UNDECLARED_ALLERGEN" | "NONE",
      "allergen_category":   e.g. "Milk",
      "original_allergen_declarations": <original string>,
      "modified_allergen_declarations": <new string>,
      "original_raw_label_text":        <original string>,   # only set if label changed
      "modified_raw_label_text":        <new string>,        # only set if label changed
      "explanation":         human-readable description of the contradiction
  }

Usage:
    python inject_allergen_contradictions.py <input.jsonl>

Output:
    <stem>_allergen_contradictions.jsonl
"""

import copy
import json
import os
import random
import re
import sys

# ---------------------------------------------------------------------------
# Allergen knowledge base
# ---------------------------------------------------------------------------

# Each allergen category:
#   "fssai_label"  : how it appears in the CONTAINS declaration
#   "en_tag"       : the en: prefix tag used in allergen_declarations field
#   "trigger_words": words in ingredient list that trigger this allergen
#   "exemptions"   : ingredient substrings that are FSSAI-exempt (no allergen needed)
#   "ghost_ingredient": a representative ingredient to INJECT into label text
#                       when creating an UNDECLARED_ALLERGEN contradiction

ALLERGENS = {
    "Gluten": {
        "fssai_label": "Gluten (Wheat)",
        "en_tags": ["en:gluten", "en:wheat"],
        "trigger_words": [
            "wheat",
            "maida",
            "atta",
            "rye",
            "barley",
            "oats",
            "spelt",
            "wheat flour",
            "wheat starch",
            "wheat gluten",
            "semolina",
            "durum",
        ],
        "exemptions": [
            "wheat based glucose",
            "wheat based dextrose",
            "wheat based maltodextrin",
            "maltodextrin",
            "glucose syrup",
            "barley glucose",
            "distillate",
            "ethyl alcohol",
            "wheat germ oil",
            "wheat bran oil",
        ],
        "ghost_ingredient": "WHEAT FLOUR",
    },
    "Milk": {
        "fssai_label": "Milk",
        "en_tags": ["en:milk", "en:dairy", "en:lactose"],
        "trigger_words": [
            "milk",
            "dairy",
            "whey",
            "lactose",
            "casein",
            "caseinate",
            "cream",
            "butter",
            "cheese",
            "ghee",
            "curd",
            "yoghurt",
            "yogurt",
            "skimmed milk",
            "milk solids",
            "milk powder",
            "milk fat",
        ],
        "exemptions": ["milk thistle"],
        "ghost_ingredient": "MILK SOLIDS",
    },
    "Egg": {
        "fssai_label": "Egg",
        "en_tags": ["en:egg", "en:eggs"],
        "trigger_words": [
            "egg",
            "albumin",
            "mayonnaise",
            "egg powder",
            "egg yolk",
            "egg white",
            "whole egg",
            "dried egg",
        ],
        "exemptions": ["egg fruit", "eggplant"],
        "ghost_ingredient": "EGG POWDER",
    },
    "Fish": {
        "fssai_label": "Fish",
        "en_tags": ["en:fish"],
        "trigger_words": [
            "fish",
            "salmon",
            "tuna",
            "cod",
            "anchovy",
            "sardine",
            "tilapia",
            "hilsa",
            "rohu",
            "mackerel",
            "fish sauce",
            "fish paste",
            "fish oil",
        ],
        "exemptions": ["fish gelatin", "fish collagen"],
        "ghost_ingredient": "FISH SAUCE",
    },
    "Nut": {
        "fssai_label": "Nut",
        "en_tags": [
            "en:nuts",
            "en:peanuts",
            "en:tree-nuts",
            "en:almonds",
            "en:cashews",
            "en:walnuts",
        ],
        "trigger_words": [
            "peanut",
            "groundnut",
            "almond",
            "walnut",
            "cashew",
            "pistachio",
            "hazelnut",
            "pecan",
            "macadamia",
            "brazil nut",
            "pine nut",
            "nut",
            "mixed nuts",
        ],
        "exemptions": [
            "coconut",
            "nutmeg",
            "butter nut squash",
            "groundnut oil",
            "peanut oil",
            "refined groundnut oil",
            "refined peanut oil",
        ],
        "ghost_ingredient": "ALMOND POWDER",
    },
    "Soy": {
        "fssai_label": "Soy",
        "en_tags": ["en:soybeans", "en:soy", "en:soya"],
        "trigger_words": [
            "soy",
            "soya",
            "soybean",
            "tofu",
            "tempeh",
            "edamame",
            "miso",
            "soy lecithin",
            "soya lecithin",
            "soy protein",
            "textured vegetable protein",
            "soya flour",
        ],
        "exemptions": [
            "soy sauce (naturally fermented)",
            "refined soy oil",
            "refined soya oil",
            "fully refined soybean oil",
        ],
        "ghost_ingredient": "SOYA FLOUR",
    },
    "Crustacean": {
        "fssai_label": "Crustacean",
        "en_tags": ["en:crustaceans", "en:shrimp", "en:prawns"],
        "trigger_words": [
            "shrimp",
            "prawn",
            "crab",
            "lobster",
            "crayfish",
            "krill",
            "barnacle",
            "crustacean",
        ],
        "exemptions": [],
        "ghost_ingredient": "SHRIMP EXTRACT",
    },
    "Sulphite": {
        "fssai_label": "Sulphite",
        "en_tags": ["en:sulphur-dioxide-and-sulphites", "en:sulphites", "en:sulfites"],
        "trigger_words": [
            "sulphite",
            "sulfite",
            "sulphur dioxide",
            "sulfur dioxide",
            "sodium metabisulphite",
            "potassium metabisulphite",
            "sodium bisulphite",
            "potassium bisulphite",
            "ins 220",
            "ins 221",
            "ins 222",
            "ins 223",
            "ins 224",
            "ins 225",
            "ins 226",
            "ins 227",
            "ins 228",
            "220",
            "221",
            "222",
            "223",
            "224",
        ],
        "exemptions": [],
        "ghost_ingredient": "SODIUM METABISULPHITE",
    },
}

# Convenience: all en: tags -> category name
_TAG_TO_CATEGORY = {}
for cat, info in ALLERGENS.items():
    for tag in info["en_tags"]:
        _TAG_TO_CATEGORY[tag.lower()] = cat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_lower(text: str) -> str:
    return text.lower()


def _is_exempt(text_lower: str, category: str) -> bool:
    """Return True if every allergen hit in the text is covered by an exemption."""
    exemptions = ALLERGENS[category]["exemptions"]
    triggers = ALLERGENS[category]["trigger_words"]
    hits = [t for t in triggers if t in text_lower]
    if not hits:
        return False
    # If ALL hits are inside an exempt phrase, treat as exempt
    non_exempt = []
    for hit in hits:
        covered = any(
            ex in text_lower for ex in exemptions if hit in ex or ex in text_lower
        )
        if not covered:
            non_exempt.append(hit)
    return len(non_exempt) == 0


def _present_in_label(text_lower: str, category: str) -> bool:
    """True if a non-exempt allergen trigger for `category` exists in label."""
    triggers = ALLERGENS[category]["trigger_words"]
    exemptions = ALLERGENS[category]["exemptions"]
    for t in triggers:
        if t in text_lower:
            # Check whether this hit is inside an exempt phrase in the actual text
            for ex in exemptions:
                if ex in text_lower and t in ex:
                    break  # this trigger is part of an exempt phrase — skip
            else:
                return True
    return False


def _declared_categories(allergen_str: str) -> set:
    """
    Parse allergen_declarations string and return a set of category names.
    Handles en: tags and free-text labels.
    """
    if not allergen_str or allergen_str.strip().lower() in (
        "no allergens listed",
        "no dietary labels",
        "en:none",
        "",
    ):
        return set()

    found = set()
    lower = allergen_str.lower()

    # Match en: tags
    for tag in _TAG_TO_CATEGORY:
        if tag in lower:
            found.add(_TAG_TO_CATEGORY[tag])

    # Match free-text category names
    for cat in ALLERGENS:
        if cat.lower() in lower:
            found.add(cat)

    return found


def _add_declaration(allergen_str: str, category: str) -> str:
    """
    Append a new allergen category to the declaration string.
    Uses the en: tag format to stay consistent.
    """
    tag = ALLERGENS[category]["en_tags"][0]
    if not allergen_str or allergen_str.strip().lower() in (
        "no allergens listed",
        "no dietary labels",
        "en:none",
        "",
    ):
        return tag
    return allergen_str.rstrip(",") + "," + tag


def _remove_declaration(allergen_str: str, category: str) -> str:
    """
    Remove all en: tags for `category` from the declaration string.
    Falls back to 'No allergens listed' if nothing remains.
    """
    tags = ALLERGENS[category]["en_tags"]
    parts = [p.strip() for p in allergen_str.split(",")]
    remaining = []
    for p in parts:
        if any(tag.lower() in p.lower() for tag in tags):
            continue
        if p.lower() in (cat.lower() for cat in ALLERGENS):
            if p.lower() == category.lower():
                continue
        remaining.append(p)
    cleaned = ", ".join(remaining).strip(", ")
    return cleaned if cleaned else "No allergens listed"


def _inject_ingredient(label_text: str, ingredient: str) -> str:
    """
    Splice `ingredient` into the ingredient list near the end,
    before any trailing CONTAINS / flavour / colour statement.
    """
    # Try to insert before "CONTAINS" statement
    contains_match = re.search(
        r"(CONTAINS\b|Contains\b|PERMITTED\b|ADDED FLAVOUR|ADDED COLOR)", label_text
    )
    if contains_match:
        pos = contains_match.start()
        before = label_text[:pos].rstrip(" ,\r\n")
        after = label_text[pos:]
        return before + ", " + ingredient + " " + after

    # Otherwise append at the end
    return label_text.rstrip(" ,\r\n.") + ", " + ingredient


def _remove_ingredient(label_text: str, category: str) -> tuple:
    """
    Remove the first non-exempt allergen ingredient for `category`
    from label_text. Returns (new_text, removed_phrase) or (original, None).
    """
    triggers = ALLERGENS[category]["trigger_words"]
    exemptions = ALLERGENS[category]["exemptions"]

    # Sort triggers by length descending so we match the longest phrase first
    for trigger in sorted(triggers, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(trigger) + r"\b", re.IGNORECASE)
        m = pattern.search(label_text)
        if not m:
            continue

        # Check exemption
        ctx = label_text[max(0, m.start() - 30) : m.end() + 30].lower()
        if any(ex in ctx for ex in exemptions):
            continue

        # Found a removable hit — try to remove the whole token/phrase around it
        # Expand to grab surrounding ingredient phrase (up to nearby comma)
        start = m.start()
        end = m.end()

        # Expand right to end of this ingredient token (stop at comma/bracket)
        right = re.match(r"[^,\[\]()\r\n]*", label_text[end:])
        if right:
            end += len(right.group(0).rstrip())

        # Expand left over non-comma chars
        left_part = label_text[:start]
        left_match = re.search(r"[^,\[\]()\r\n]*$", left_part)
        if left_match:
            remove_start = start - len(left_match.group(0).lstrip())
        else:
            remove_start = start

        removed = label_text[remove_start:end]

        # Clean up the surrounding comma/space
        new_text = label_text[:remove_start].rstrip(", ") + label_text[end:].lstrip(
            ", "
        )
        new_text = re.sub(r",\s*,", ",", new_text)
        return new_text, removed.strip()

    return label_text, None


# ---------------------------------------------------------------------------
# Core contradiction logic
# ---------------------------------------------------------------------------


def inject_contradiction(record: dict, rng: random.Random) -> dict:
    rec = copy.deepcopy(record)
    rec.pop("allergen_contradiction_injected", None)
    rec.pop("allergen_contradiction_changes", None)

    label_text = rec.get("raw_label_text", "")
    allergen_str = rec.get("allergen_declarations", "No allergens listed")
    label_lower = _label_lower(label_text)

    declared_cats = _declared_categories(allergen_str)

    # Identify which allergens ARE present in label (non-exempt)
    present_cats = set(cat for cat in ALLERGENS if _present_in_label(label_lower, cat))

    # Identify which allergens are declared but NOT present in label
    ghost_candidates = declared_cats - present_cats

    # Identify which allergens are present in label but NOT declared
    undeclared_candidates = present_cats - declared_cats

    # Identify allergens not present & not declared (for ghost injection)
    absent_cats = set(ALLERGENS.keys()) - present_cats - declared_cats

    # -------------------------------------------------------------------
    # Decide which contradiction type to attempt
    # -------------------------------------------------------------------

    # Build a weighted pool of possible actions
    # Weight towards whichever type has good candidates
    actions = []

    # TYPE A: GHOST_ALLERGEN — declare an allergen whose ingredient isn't in label
    # Sub-type A1: pick from absent_cats (clean ghost — add to declaration only)
    if absent_cats:
        actions.append(("GHOST_ALLERGEN_ADD", sorted(absent_cats)))

    # Sub-type A2: remove an ingredient that backs an existing declaration
    if ghost_candidates:
        actions.append(("GHOST_ALLERGEN_REMOVE_INGREDIENT", sorted(ghost_candidates)))

    # TYPE B: UNDECLARED_ALLERGEN
    # Sub-type B1: add an ingredient to label without declaring it
    if absent_cats:
        actions.append(("UNDECLARED_ADD_INGREDIENT", sorted(absent_cats)))

    # Sub-type B2: strip an existing declaration while keeping ingredient in label
    if undeclared_candidates:
        actions.append(("UNDECLARED_REMOVE_DECLARATION", sorted(undeclared_candidates)))

    if not actions:
        # No contradiction possible — label is empty/garbled
        rec["allergen_contradiction_injected"] = False
        rec["allergen_contradiction_changes"] = {
            "type": "NONE",
            "explanation": "Label text too sparse to inject a meaningful contradiction.",
        }
        return rec

    # Prefer TYPE B (more realistic on-label contradiction) when possible
    b_actions = [a for a in actions if a[0].startswith("UNDECLARED")]
    chosen_pool = b_actions if b_actions and rng.random() < 0.6 else actions
    action_type, candidates = rng.choice(chosen_pool)
    category = rng.choice(candidates)

    orig_label = label_text
    orig_allergen = allergen_str
    changes = {
        "type": action_type,
        "allergen_category": category,
        "fssai_declaration_label": ALLERGENS[category]["fssai_label"],
        "original_allergen_declarations": orig_allergen,
        "modified_allergen_declarations": allergen_str,
        "original_raw_label_text": orig_label,
        "modified_raw_label_text": label_text,
        "explanation": "",
    }

    # -------------------------------------------------------------------
    # Apply the chosen action
    # -------------------------------------------------------------------

    if action_type == "GHOST_ALLERGEN_ADD":
        # Add allergen to declaration but ingredient is NOT in label
        new_allergen = _add_declaration(allergen_str, category)
        rec["allergen_declarations"] = new_allergen
        changes["modified_allergen_declarations"] = new_allergen
        changes["explanation"] = (
            f"GHOST ALLERGEN: '{ALLERGENS[category]['fssai_label']}' added to "
            f"allergen_declarations (CONTAINS declaration) but no corresponding "
            f"ingredient is present in raw_label_text. "
            f"Per FSSAI reg 2.2.1(14), this is a false positive allergen declaration."
        )

    elif action_type == "GHOST_ALLERGEN_REMOVE_INGREDIENT":
        # Remove from label the ingredient that backs a declared allergen
        new_label, removed = _remove_ingredient(label_text, category)
        if removed is None:
            # fallback: just add a ghost declaration instead
            absent_fallback = (
                sorted(absent_cats)
                if absent_cats
                else sorted(ALLERGENS.keys() - declared_cats)
            )
            if not absent_fallback:
                rec["allergen_contradiction_injected"] = False
                rec["allergen_contradiction_changes"] = {
                    "type": "NONE",
                    "explanation": "No actionable contradiction found.",
                }
                return rec
            category = rng.choice(absent_fallback)
            new_allergen = _add_declaration(allergen_str, category)
            rec["allergen_declarations"] = new_allergen
            changes["type"] = "GHOST_ALLERGEN_ADD"
            changes["allergen_category"] = category
            changes["modified_allergen_declarations"] = new_allergen
            changes["explanation"] = (
                f"GHOST ALLERGEN: '{ALLERGENS[category]['fssai_label']}' added to "
                f"allergen_declarations but no ingredient for it exists in raw_label_text."
            )
        else:
            rec["raw_label_text"] = new_label
            changes["modified_raw_label_text"] = new_label
            changes["explanation"] = (
                f"GHOST ALLERGEN: '{category}' is declared in allergen_declarations "
                f"(CONTAINS {ALLERGENS[category]['fssai_label']}) but the triggering "
                f"ingredient '{removed}' has been removed from raw_label_text. "
                f"The label now falsely declares an allergen with no supporting ingredient."
            )

    elif action_type == "UNDECLARED_ADD_INGREDIENT":
        # Add allergenic ingredient to label without declaring it
        ingredient = ALLERGENS[category]["ghost_ingredient"]
        new_label = _inject_ingredient(label_text, ingredient)
        rec["raw_label_text"] = new_label
        changes["modified_raw_label_text"] = new_label
        changes["explanation"] = (
            f"UNDECLARED ALLERGEN: '{ingredient}' injected into raw_label_text "
            f"but '{ALLERGENS[category]['fssai_label']}' is NOT added to "
            f"allergen_declarations. Per FSSAI reg 2.2.1(14), this ingredient "
            f"requires a mandatory CONTAINS {ALLERGENS[category]['fssai_label']} declaration."
        )

    elif action_type == "UNDECLARED_REMOVE_DECLARATION":
        # Remove the allergen declaration while the ingredient remains in label
        new_allergen = _remove_declaration(allergen_str, category)
        rec["allergen_declarations"] = new_allergen
        changes["modified_allergen_declarations"] = new_allergen
        changes["explanation"] = (
            f"UNDECLARED ALLERGEN: '{category}' ingredient is present in "
            f"raw_label_text but its declaration has been removed from "
            f"allergen_declarations. Per FSSAI reg 2.2.1(14), a mandatory "
            f"CONTAINS {ALLERGENS[category]['fssai_label']} declaration is required."
        )

    rec["allergen_contradiction_injected"] = True
    rec["allergen_contradiction_changes"] = changes
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
    out_path = stem + "_allergen_contradictions.jsonl"

    injected = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for rec in records:
            processed = inject_contradiction(rec, rng)
            out_f.write(json.dumps(processed, ensure_ascii=False) + "\n")
            if processed["allergen_contradiction_injected"]:
                injected += 1
            else:
                skipped += 1

    print(f"  Contradictions injected : {injected}")
    print(f"  Could not inject        : {skipped}")
    print(f"  Output written to       : {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inject_allergen_contradictions.py <input.jsonl>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)
    process_file(path)

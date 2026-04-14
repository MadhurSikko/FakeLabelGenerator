"""
inject_ins_errors.py
--------------------
Takes a JSONL file of food-product records and produces one output JSONL
where INS codes found in `raw_label_text` are replaced with codes from a
*different* functional category, introducing detectable FSSAI violations.

Error-injection strategy
------------------------
Case 1 – Named functional class outside the brackets
  e.g. "emulsifier (471, 472e)"
  → ALL codes inside are replaced with codes from a DIFFERENT family.
    The label now claims e.g. "emulsifier (202, 211)" which violates FSSAI
    Schedule V: those codes are preservatives, not emulsifiers.

Case 2 – No named class outside the brackets (bare group or explicit INS/E prefix)
  e.g. "contains (330, 331, 471)"  or  "INS 330, INS 471"
  → Codes are replaced so that each code in the group comes from a
    DIFFERENT family than its neighbour(s).  The resulting group therefore
    mixes incompatible functional classes, which an LLM or compliance
    checker should flag as anomalous.

Both cases produce violations that are clearly detectable:
  • A colour code inside a bracket labelled "preservative" is illegal.
  • A group whose members span antioxidant / colour / mineral-salt families
    with no functional-class label is suspicious / non-compliant.

Only `raw_label_text` is modified. All other fields are untouched.

Two new keys are added to every record:
  "ins_code_found" : true  – at least one INS code detected & swapped
                   : false – no INS code found
  "ins_changes"    : list of change descriptors (empty when false)
    {
      "original_code":    "503(ii)",
      "replacement_code": "202",
      "original_name":    "ammonium bicarbonate",
      "replacement_name": "potassium sorbate",
      "original_category":  "mineral salt",
      "injected_category":  "preservative",
      "violation":          "category_mismatch",
      "violation_detail":   "label claims mineral salt but code belongs to preservative",
      "char_index":         114
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


def replacement_from_different_family(
    norm_key: str,
    rng: "random.Random",
    exclude_categories: "set | None" = None,
) -> "tuple | None":
    """
    Pick a replacement code from a DIFFERENT functional category than norm_key.

    exclude_categories — additional families to avoid (used when building a
    group whose members each come from a distinct family).

    Returns (new_code, new_name, orig_primary_cat, injected_cat) or None.
    """
    if norm_key not in INS_DB:
        return None
    _, orig_categories = INS_DB[norm_key]
    orig_cat_set = set(orig_categories)
    forbidden = orig_cat_set | (exclude_categories or set())

    available_cats = [c for c in CATEGORY_TO_CODES if c not in forbidden]
    rng.shuffle(available_cats)

    for cat in available_cats:
        candidates = CATEGORY_TO_CODES[cat]
        if candidates:
            new_code = rng.choice(candidates)
            new_name, _ = INS_DB[new_code]
            orig_cat = next(iter(orig_categories))
            return new_code, new_name, orig_cat, cat

    return None


# Legacy same-family helper retained for reference / unit tests
def replacement_for(norm_key: str, rng: "random.Random") -> "tuple | None":
    """
    (Legacy) Pick a replacement from the SAME functional category.
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


# ---------------------------------------------------------------------------
# Group detection — identify bracket groups and their declared functional class
# ---------------------------------------------------------------------------

# Maps canonical class-title keyword to its INS family name in CATEGORY_TO_CODES
_TITLE_TO_FAMILY = {
    "colour": "colour",
    "color": "colour",
    "preservative": "preservative",
    "antioxidant": "antioxidant",
    "emulsifier": "emulsifier",
    "stabiliser": "thickener",
    "stabilizer": "thickener",
    "thickener": "thickener",
    "sequestrant": "sequestrant",
    "acidity regulator": "acidity regulator",
    "raising agent": "mineral salt",
    "anti-caking agent": "anti-caking agent",
    "anticaking agent": "anti-caking agent",
    "anti caking agent": "anti-caking agent",
    "flavour enhancer": "flavour enhancer",
    "flavor enhancer": "flavour enhancer",
    "humectant": "sweetener",
    "sweetener": "sweetener",
    "flour treatment agent": "flour treatment agent",
    "firming agent": "mineral salt",
    "bleaching agent": "flour treatment agent",
    "sequesterant": "sequestrant",
}

_CLASS_TITLE_CAPTURE = re.compile(
    r"\b(colour|color|preservative|antioxidant|emulsifier|"
    r"stabilisers?|stabilizers?|thickeners?|sequestrants?|"
    r"acidity\s+regulator|raising\s+agent|anti.?caking(?:\s+agent)?|"
    r"anticaking(?:\s+agent)?|flavou?r\s+enhancer|humectant|sweeteners?|"
    r"flour\s+treatment(?:\s+agent)?|firming\s+agent|bleaching\s+agent|"
    r"sequesterants?)\b",
    re.IGNORECASE,
)


def _declared_family_before(text: str, group_start: int) -> str | None:
    """
    Return the INS-DB family name if a functional-class title appears within
    60 characters before `group_start`, otherwise None.
    """
    context = text[max(0, group_start - 60) : group_start]
    m = _CLASS_TITLE_CAPTURE.search(context)
    if not m:
        return None
    keyword = re.sub(r"\s+", " ", m.group(1).lower().rstrip("s"))
    return _TITLE_TO_FAMILY.get(keyword)


def _group_spans(text: str) -> list:
    """
    Return list of (group_start, group_end, [token_positions]) for every
    bracket group detected by _BARE_GROUP.  token_positions are
    (start, end, raw, norm) tuples for each code inside the group.
    """
    groups = []
    for m in _BARE_GROUP.finditer(text):
        group_content = m.group(1)
        context_before = text[max(0, m.start() - 60) : m.start()]
        has_class = bool(_CLASS_TITLES.search(context_before))
        tokens_raw = re.findall(_CODE_PAT, group_content)
        has_multiple = len(tokens_raw) >= 2
        if not (has_class or has_multiple):
            continue

        token_positions = []
        search_from = m.start(1)
        for tok in tokens_raw:
            pos = text.find(tok, search_from)
            if pos != -1:
                norm = normalise_code(tok)
                token_positions.append((pos, pos + len(tok), tok, norm))
                search_from = pos + len(tok)

        if token_positions:
            groups.append((m.start(), m.end(), token_positions))

    return groups


# ---------------------------------------------------------------------------
# Core record processing — cross-family injection
# ---------------------------------------------------------------------------


def _violation_detail(
    declared_family: str | None, orig_cat: str, injected_cat: str
) -> str:
    if declared_family:
        return (
            f"label declares '{declared_family}' but replacement code "
            f"belongs to '{injected_cat}' (original was '{orig_cat}')"
        )
    return (
        f"group mixes '{orig_cat}' and '{injected_cat}' — "
        "different functional classes in one unlabelled group"
    )


def process_record(record: dict, rng: random.Random) -> dict:
    """
    Inject cross-family INS errors so mismatches are detectable by an LLM.

    Strategy
    --------
    For each bracket group found in the text:

      Case 1 – declared functional class in the 60 chars before the bracket
        Replace EVERY code in the group with a code from a DIFFERENT family
        (any family other than the declared one AND the original family).
        → e.g. "emulsifier (471, 472e)"  becomes  "emulsifier (202, 211)"
          Those are preservatives — a clear FSSAI Schedule V violation.

      Case 2 – no declared class (bare group or multiple INS-prefixed codes)
        Replace codes so that consecutive codes come from DIFFERENT families,
        ensuring the group spans at least two distinct functional classes.
        → e.g. "(330, 331, 471)"  becomes  "(330, 202, 471)"
          Now the group contains an acidity regulator AND a preservative.

    Explicit INS/E-prefixed codes that do NOT fall inside a detected group
    are treated as Case 2 (single-token groups).
    """
    rec = copy.deepcopy(record)
    rec.pop("ins_code_found", None)
    rec.pop("ins_changes", None)

    text = rec.get("raw_label_text", "")

    # Collect all tokens and which bracket group they belong to
    groups = _group_spans(text)

    # Build a set of positions already covered by a group
    group_covered: dict[
        int, tuple
    ] = {}  # token_start -> (group_start, declared_family)
    for g_start, g_end, tok_positions in groups:
        decl = _declared_family_before(text, g_start)
        for t_start, t_end, raw, norm in tok_positions:
            group_covered[t_start] = (g_start, decl)

    # Also gather explicit INS/E-prefix tokens not inside any group
    all_tokens = find_ins_tokens(text)
    lone_tokens = [
        (s, e, raw, norm)
        for s, e, raw, norm in all_tokens
        if s not in group_covered and norm in INS_DB
    ]

    # -----------------------------------------------------------------------
    # Build replacement plan: {token_start: (new_code, new_name, orig_cat, injected_cat)}
    # -----------------------------------------------------------------------
    plan: dict[int, tuple] = {}

    # ---- Case 1 & 2: bracket groups ----
    for g_start, g_end, tok_positions in groups:
        decl = _declared_family_before(text, g_start)

        if decl:
            # Case 1 — replace ALL tokens with codes from a family != declared AND != original
            for t_start, t_end, raw, norm in tok_positions:
                if norm not in INS_DB:
                    continue
                result = replacement_from_different_family(
                    norm,
                    rng,
                    exclude_categories={decl},  # must differ from declared label
                )
                if result:
                    plan[t_start] = (
                        result  # (new_code, new_name, orig_cat, injected_cat)
                    )
        else:
            # Case 2 — ensure the group spans multiple families
            used_cats: set[str] = set()
            for idx, (t_start, t_end, raw, norm) in enumerate(tok_positions):
                if norm not in INS_DB:
                    continue
                # First token: pick freely from a different family
                # Subsequent tokens: also avoid families already used, to maximise diversity
                result = replacement_from_different_family(
                    norm,
                    rng,
                    exclude_categories=used_cats if idx > 0 else None,
                )
                if result:
                    plan[t_start] = result
                    used_cats.add(result[3])  # injected_cat

    # ---- Lone explicit INS/E tokens (Case 2 – single token) ----
    for s, e, raw, norm in lone_tokens:
        if s in plan:
            continue
        result = replacement_from_different_family(norm, rng)
        if result:
            plan[s] = result

    if not plan:
        rec["ins_code_found"] = False
        rec["ins_changes"] = []
        return rec

    # -----------------------------------------------------------------------
    # Apply substitutions right-to-left so offsets stay valid
    # -----------------------------------------------------------------------
    changes = []
    sorted_positions = sorted(plan.keys(), reverse=True)

    for t_start in sorted_positions:
        new_code, new_name, orig_cat, injected_cat = plan[t_start]

        # Find the raw token at this position
        matching = [(s, e, raw, norm) for s, e, raw, norm in all_tokens if s == t_start]
        if not matching:
            # Also search group tokens
            for _, _, tok_positions in groups:
                for ts, te, raw, norm in tok_positions:
                    if ts == t_start:
                        matching = [(ts, te, raw, norm)]
                        break
        if not matching:
            continue

        _, t_end, raw, norm = matching[0]
        orig_name, _ = INS_DB[norm]

        current_text = rec["raw_label_text"]
        window_start = max(0, t_start - 5)
        actual_pos = current_text.find(raw, window_start)
        if actual_pos == -1:
            continue

        rec["raw_label_text"] = (
            current_text[:actual_pos] + new_code + current_text[actual_pos + len(raw) :]
        )

        # Determine declared family (for violation detail message)
        group_info = group_covered.get(t_start)
        decl = group_info[1] if group_info else None

        changes.insert(
            0,
            {
                "original_code": norm,
                "replacement_code": new_code,
                "original_name": orig_name,
                "replacement_name": new_name,
                "original_category": orig_cat,
                "injected_category": injected_cat,
                "violation": "category_mismatch",
                "violation_detail": _violation_detail(decl, orig_cat, injected_cat),
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

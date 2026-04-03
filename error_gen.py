import json
import random
import string


def apply_single_typo(text):
    """Introduces 1 typo, ensuring we don't accidentally create 'stealth' valid numbers."""
    chars = list(text)
    if len(chars) < 2:
        return text

    decimal_indices = [i for i, char in enumerate(chars) if char == "."]

    options = ["swap", "delete", "replace"]
    if decimal_indices:
        # Give a high chance to drop the decimal (tests the LLM's math validation)
        options.extend(["decimal_drop", "decimal_drop"])

    typo_type = random.choice(options)

    if typo_type == "decimal_drop":
        # Safely drop the decimal so 3.5g becomes 35g (triggering a math anomaly)
        idx = random.choice(decimal_indices)
        chars.pop(idx)
        return "".join(chars)

    idx = random.randint(0, len(chars) - 1)

    if typo_type == "swap" and idx < len(chars) - 1:
        char1, char2 = chars[idx], chars[idx + 1]

        # PREVENT STEALTH NUMBERS: Don't swap digits/decimals with each other (avoids 3.5 -> .35)
        if (char1.isdigit() or char1 == ".") and (char2.isdigit() or char2 == "."):
            # Force a syntax error instead
            chars[idx] = random.choice(string.ascii_lowercase)
        else:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    elif typo_type == "delete" and len(chars) > 2:
        # PREVENT STEALTH NUMBERS: Don't delete a digit (avoids 12g -> 1g)
        if chars[idx].isdigit():
            # Force a syntax error instead
            chars[idx] = random.choice(string.ascii_lowercase)
        else:
            chars.pop(idx)

    else:
        # For 'replace', always use a letter. If it hits a number, it creates an obvious typo.
        chars[idx] = random.choice(string.ascii_lowercase)

    return "".join(chars)


def distribute_typos_in_entry(entry, num_errors):
    """Randomly distributes `num_errors` between raw text and nutrition facts."""
    for _ in range(num_errors):
        targets = []

        # 1. Check if raw_label_text is available to be mangled
        raw_text = entry.get("raw_label_text", "")
        if raw_text and raw_text != "NO_INGREDIENTS" and len(raw_text) >= 2:
            targets.append("raw_label")

        # 2. Check if there are valid nutrition facts to mangle
        nutrition = entry.get("nutrition_facts", {})
        valid_nutri_keys = []
        if isinstance(nutrition, dict):
            # Only pick nutrition values that are strings and long enough to edit (ignoring "N/A")
            valid_nutri_keys = [
                k
                for k, v in nutrition.items()
                if isinstance(v, str) and len(v) >= 2 and v != "N/A"
            ]
            if valid_nutri_keys:
                targets.append("nutrition")

        # If there's absolutely no text to modify in this entry, skip it
        if not targets:
            break

        # 3. Flip a coin to pick where this specific error goes
        chosen_target = random.choice(targets)

        if chosen_target == "raw_label":
            entry["raw_label_text"] = apply_single_typo(entry["raw_label_text"])
        elif chosen_target == "nutrition":
            # Pick a random nutrient (e.g., "carbohydrates") and mangle its value
            key_to_mangle = random.choice(valid_nutri_keys)
            entry["nutrition_facts"][key_to_mangle] = apply_single_typo(
                entry["nutrition_facts"][key_to_mangle]
            )

    return entry


def create_typo_datasets(input_filename):
    configs = {
        1: "dataset_1_error.jsonl",
        2: "dataset_2_error.jsonl",
        3: "dataset_3_error.jsonl",
    }

    try:
        with open(input_filename, "r", encoding="utf-8") as f:
            original_lines = f.readlines()
    except FileNotFoundError:
        print(f"[!] Could not find {input_filename}. Make sure the file exists.")
        return

    for num_errors, output_filename in configs.items():
        print(f"Generating {output_filename}...")

        with open(output_filename, "w", encoding="utf-8") as out_f:
            for line in original_lines:
                entry = json.loads(line)

                # Apply the distributed typos
                mangled_entry = distribute_typos_in_entry(entry, num_errors)

                out_f.write(json.dumps(mangled_entry, ensure_ascii=False) + "\n")

        print(
            f"[+] Successfully saved {len(original_lines)} entries to {output_filename}"
        )


# --- Execution ---
INPUT_FILE = "fssai_benchmark_dataset.jsonl"
create_typo_datasets(INPUT_FILE)

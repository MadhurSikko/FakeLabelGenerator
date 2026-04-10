from datetime import datetime

import requests

CUTOFF_YEAR = 2024


def get_diverse_reliable_2024_barcodes(target_count=50):
    print(
        f"Fetching {target_count} recent, highly reliable, diverse Indian products from Open Food Facts..."
    )

    # Using the OFF search API, targeting India, sorted by newest edits
    url = "https://in.openfoodfacts.org/cgi/search.pl"
    params = {
        "action": "process",
        "sort_by": "popularity",
        "page_size": 1000,  # Increased to 1000 because strict reliability filters will reject many products
        "json": "true",
    }
    headers = {"User-Agent": "FSSAI_AI_Benchmark_Research - Python - Version 1.5"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        products = response.json().get("products", [])

        valid_barcodes = []
        category_tracker = {}

        for p in products:
            if len(valid_barcodes) >= target_count:
                break

            barcode = p.get("code")
            if not barcode:
                continue

            # 1. Enforce the 2024 Cutoff
            last_modified_unix = p.get("last_modified_t", 0)
            last_modified_year = (
                datetime.fromtimestamp(last_modified_unix).year
                if last_modified_unix
                else 0
            )
            if last_modified_year < CUTOFF_YEAR:
                continue

            # 2. Basic Data Presence Check
            if not p.get("ingredients_text") or not p.get("nutriments"):
                continue

            # --- NEW: STRICT RELIABILITY CHECKS ---
            states = p.get("states_tags", [])

            # 3. Check OFF Completion States
            # Ensure the community or manufacturer has marked these sections as fully complete
            if (
                "en:nutrition-facts-completed" not in states
                or "en:ingredients-completed" not in states
            ):
                continue

            # 4. Check for Data Quality Errors
            # OFF runs automated checks (e.g., nutrition values adding up to > 100g). Exclude if it fails.
            errors = p.get("data_quality_errors_tags", [])
            if errors:
                continue

            # 5. Require Photographic Proof (Highly Recommended for Benchmarks)
            # This ensures there is a picture of the label to back up the text data.
            if not p.get("image_ingredients_url") or not p.get("image_nutrition_url"):
                continue
            # --------------------------------------

            # 6. Enforce Variety (Limit to 3 items per main category)
            main_category = p.get("main_category", "unknown")
            if (
                category_tracker.get(main_category, 0) >= 3
                and main_category != "unknown"
            ):
                continue

            # If it passes all strict checks, add it to our list!
            valid_barcodes.append(barcode)
            category_tracker[main_category] = category_tracker.get(main_category, 0) + 1

        return valid_barcodes

    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching barcodes: {e}")
        return []


# --- Run the fetcher ---
fresh_barcodes = get_diverse_reliable_2024_barcodes(50)
print(
    f"\nSuccessfully gathered {len(fresh_barcodes)} highly reliable barcodes updated in {CUTOFF_YEAR} or later."
)
print("Copy this list into your main script:\n")
print(fresh_barcodes)

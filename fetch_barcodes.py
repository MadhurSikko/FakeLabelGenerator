from datetime import datetime

import requests

CUTOFF_YEAR = 2024


def get_diverse_2024_barcodes(target_count=50):
    print(
        f"Fetching {target_count} recent, diverse Indian products from Open Food Facts..."
    )

    # Using the OFF search API, targeting India, sorted by newest edits
    url = "https://in.openfoodfacts.org/cgi/search.pl"
    params = {
        "action": "process",
        "sort_by": "popularity",  # <-- Sorts by most scanned/famous products first
        "page_size": 500,  # <-- Increased to 500 so we have a bigger pool to filter
        "json": "true",
    }
    headers = {"User-Agent": "FSSAI_AI_Benchmark_Research - Python - Version 1.4"}

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

            # 1. Enforce the 2024 Cutoff
            last_modified_unix = p.get("last_modified_t", 0)
            last_modified_year = (
                datetime.fromtimestamp(last_modified_unix).year
                if last_modified_unix
                else 0
            )
            if last_modified_year < CUTOFF_YEAR:
                continue

            # 2. Enforce Data Quality (Must have ingredients and nutrition facts for your LLM)
            if not p.get("ingredients_text") or not p.get("nutriments"):
                continue

            # 3. Enforce Variety (Limit to 3 items per main category)
            main_category = p.get("main_category", "unknown")
            if (
                category_tracker.get(main_category, 0) >= 3
                and main_category != "unknown"
            ):
                continue

            # If it passes all checks, add it to our list!
            valid_barcodes.append(barcode)
            category_tracker[main_category] = category_tracker.get(main_category, 0) + 1

        return valid_barcodes

    except requests.exceptions.RequestException as e:
        print(f"[!] Error fetching barcodes: {e}")
        return []


# --- Run the fetcher ---
fresh_barcodes = get_diverse_2024_barcodes(50)
print(
    f"\nSuccessfully gathered {len(fresh_barcodes)} valid barcodes updated in {CUTOFF_YEAR} or later."
)
print("Copy this list into your main script:\n")
print(fresh_barcodes)

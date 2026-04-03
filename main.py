import json
import time
from datetime import datetime

import requests

# Insert your 50 generated barcodes here
barcodes = [
    "8901719134845",
    "8901719134852",
    "8901063139329",
    "8902080000227",
    "8906010500764",
    "8901063093522",
    "8901058017687",
    "8901063162914",
    "8906010500559",
    "8901764042911",
    "8906010502232",
    "8906010500900",
    "8906010500023",
    "8901764032912",
    "7622202225512",
    "8904043901015",
    "8901058000290",
    "3948764032912",
    "8901063092853",
    "8906017290040",
    "8901719129988",
    "8901030921667",
    "8901764092206",
    "8901030897542",
    "8901764082405",
    "8906010500337",
    "8901063029255",
    "8901491100519",
    "8901262010016",
    "8901262200196",
    "8901764032707",
    "8906010502294",
    "8906010500078",
    "8901491366052",
    "8901719135248",
    "8906010501570",
    "8996001312506",
    "8901058005233",
    "8902080104581",
    "8901063023901",
    "8901030921797",
    "8901719135118",
    "3948764061257",
    "89080153",
    "8901764032905",
    "8904272600291",
    "8901764092305",
    "8901595862962",
    "8901262260121",
    "8901491361026",
]

CUTOFF_YEAR = 2024


def fetch_product_context(barcode):
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    headers = {"User-Agent": "FSSAI_AI_Benchmark_Research - Python - Version 1.5"}

    # --- NEW: RETRY LOGIC WITH EXPONENTIAL BACKOFF ---
    max_retries = 5
    retry_delay = 5  # Start by waiting 5 seconds if we hit a 429 limit

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)

            # Catch the 429 specifically and WAIT instead of crashing
            if response.status_code == 429:
                print(
                    f"[!] 429 Too Many Requests. Pausing for {retry_delay}s (Attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(retry_delay)
                retry_delay *= (
                    2  # Double the wait time for the next attempt (5s, 10s, 20s...)
                )
                continue

            # If it's a different error (like 404 or 500), this will raise it normally
            response.raise_for_status()
            data = response.json()

            # --- DATA PARSING LOGIC ---
            if data.get("status") == 1:
                product = data.get("product", {})

                last_modified_unix = product.get("last_modified_t", 0)
                last_modified_year = (
                    datetime.fromtimestamp(last_modified_unix).year
                    if last_modified_unix
                    else 0
                )

                if last_modified_year < CUTOFF_YEAR:
                    print(f"[-] SKIPPED: {barcode} too old ({last_modified_year})")
                    return None

                nutriments = product.get("nutriments", {})

                def get_nutri(key):
                    val = nutriments.get(f"{key}_value")
                    unit = nutriments.get(f"{key}_unit", "g")
                    return f"{val} {unit}" if val is not None else "N/A"

                energy_val = nutriments.get("energy-kcal_value")
                energy_str = f"{energy_val} kcal" if energy_val is not None else "N/A"

                nutrition_data = {
                    "energy": energy_str,
                    "fat": get_nutri("fat"),
                    "saturated_fat": get_nutri("saturated-fat"),
                    "carbohydrates": get_nutri("carbohydrates"),
                    "sugars": get_nutri("sugars"),
                    "proteins": get_nutri("proteins"),
                    "sodium": get_nutri("sodium"),
                }

                ingredients = (
                    product.get("ingredients_text_en")
                    or product.get("ingredients_text")
                    or "NO_INGREDIENTS"
                )

                return {
                    "product_id": str(barcode),
                    "product_name": product.get("product_name") or "Unknown Product",
                    "fssai_category": product.get("categories") or "Uncategorized",
                    "raw_label_text": ingredients,
                    "nutrition_facts": nutrition_data,
                    "dietary_declarations": product.get("labels")
                    or "No dietary labels",
                    "allergen_declarations": product.get("allergens")
                    or "No allergens listed",
                    "data_year": last_modified_year,
                    "image_url": product.get("image_ingredients_url") or "NO_IMAGE",
                    "ground_truth_status": "PENDING_REVIEW",
                }
            else:
                print(f"[-] Product not found: {barcode}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"[!] Connection error on barcode {barcode}: {e}")
            return None  # Hard errors break out of the loop

    # If we exhaust all 5 retries
    print(
        f"[!] Failed to fetch {barcode} after {max_retries} attempts due to severe rate limits."
    )
    return None


# --- EXECUTION ---
dataset_filename = "fssai_benchmark_dataset.jsonl"

with open(dataset_filename, "a", encoding="utf-8") as f:
    for code in barcodes:
        print(f"Processing {code}...")
        entry = fetch_product_context(code)
        if entry:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
            print(f"[+] Added: {entry['product_name']}\n")

        # INCREASED BASE DELAY
        # A 1.5s delay keeps us safely at ~40 req/minute, leaving plenty of room
        # below the 100 req/minute limit in case you start/stop the script a lot.
        time.sleep(1.5)

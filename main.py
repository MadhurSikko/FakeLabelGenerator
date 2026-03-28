import json
import time
from datetime import datetime

import requests

barcodes = [
    "8901764042911",  # Thumbs up
    "8902080404094",  # Slice Mango Drink
    "8901888005748",  # Real Masala Mixed Fruit Juice – 1L
    "8906080602818",  # paper-boat-pomegranate-600ml
    "7622202334009",  # dairy-milk
    "8901058903164",  # kitkat-nestle
    "8901030897542",  # kissan-fresh-tomato
    "8901719125478",  # parle-g-gold-75g
    "8901491101844",  # Lay-s-potato-chips
    "8901491100519",  # Kurkure
    "8904004400731",  # aloo-bhujia-haldiram-s
    "8901725007447",  # original-style-potato-chips-bingo
    "8909081005046",  # dark-fantasy-choco-fills-sunfeast
    "8901725018016",  # yippee-noodles-snack-pack-sunfeast
    "8901063092853",  # good-day-cashew-cookie-britannia
]  # 15 barcodes

# Set your cutoff date (e.g., we only want data modified after Jan 1, 2024)
CUTOFF_YEAR = 2024


def fetch_product_context(barcode):
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    headers = {"User-Agent": "FSSAI_AI_Benchmark_Research - Python - Version 1.2"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == 1:
            product = data.get("product", {})

            # --- THE NEW TIMESTAMP CHECK ---
            last_modified_unix = product.get("last_modified_t", 0)

            # Convert Unix to a readable year
            last_modified_year = (
                datetime.fromtimestamp(last_modified_unix).year
                if last_modified_unix
                else 0
            )

            # If the data is too old, reject it before adding it to your dataset
            if last_modified_year < CUTOFF_YEAR:
                print(
                    f"[-] SKIPPED: Barcode {barcode} is too old (Last updated: {last_modified_year})"
                )
                return None
            # -------------------------------

            product_name = product.get("product_name", "Unknown Product")
            ingredients = product.get("ingredients_text_en") or product.get(
                "ingredients_text", "NO_INGREDIENTS"
            )
            image_url = product.get("image_ingredients_url", "NO_IMAGE")
            category_tags = product.get("categories", "Uncategorized")
            allergen_tags = product.get("allergens", "No allergens listed")
            dietary_labels = product.get("labels", "No dietary labels")

            return {
                "product_id": str(barcode),
                "product_name": product_name,
                "fssai_category": category_tags,
                "dietary_declarations": dietary_labels,
                "allergen_declarations": allergen_tags,
                "raw_label_text": ingredients,
                "data_year": last_modified_year,  # Good to keep a record of this!
                "image_url": image_url,
                "ground_truth_status": "PENDING_REVIEW",
                "ground_truth_reasoning": "",
                "flagged_ingredients": [],
            }
        else:
            print(f"[-] Product not found for barcode: {barcode}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"[!] Connection error: {e}")
        return None


dataset_filename = "fssai_benchmark_dataset.jsonl"

with open(dataset_filename, "a", encoding="utf-8") as f:
    for code in barcodes:
        print(f"Fetching data for barcode {code}...")
        entry = fetch_product_context(code)

        if entry:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
            print(
                f"[+] Added: {entry['product_name']} (Updated: {entry['data_year']})\n"
            )

        time.sleep(1)

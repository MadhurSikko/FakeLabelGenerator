import json
import os
import time

from google import genai
from google.genai import types

# 1. Initialize the Client
# Replace with your actual key or set it as an environment variable
client = genai.Client(api_key="")

# Choose the 2026 Flash model for the best balance of speed and logic
MODEL_ID = "gemini-3.1-flash-lite-preview"

INPUT_FILES = [
    "dataset_1_error.jsonl",
    "dataset_2_error.jsonl",
    "dataset_3_error.jsonl",
]


def analyze_with_gemini(entry):
    """Sends targeted data to Gemini using the modern 2026 SDK."""

    targeted_data = {
        "raw_label_text": entry.get("raw_label_text", ""),
        "nutrition_facts": entry.get("nutrition_facts", {}),
    }

    # Your updated instruction
    sys_instruction = """You are an expert FSSAI Regulatory Auditor evaluating food labels strictly against the Food Safety and Standards (Labelling and Display) Regulations, 2020.

    Context Constraint: You will be provided ONLY with the 'Nutrition Facts' and 'Ingredients' text of unknown food items. Your job is to audit this specific data for regulatory compliance, accuracy, and logical consistency. Do not flag errors for missing general packaging details (e.g., Net Weight, Expiry Date, Veg/Non-Veg logos, Manufacturer Address) as they are outside your scope. Focus purely on the nutritional panel and ingredient list.

    Audit the provided data against these specific FSSAI rules:
    1. Mandatory Nutrient Declarations: Ensure the core FSSAI mandatory fields are present: Energy (kcal), Protein (g), Carbohydrates (g), Total Sugars (g), Added Sugars (g), Total Fat (g), Saturated Fat (g), Trans Fat (g), Cholesterol (mg), and Sodium (mg).
    2. Ingredient Hierarchy & Plausibility: By FSSAI law, ingredients must be listed in descending order by weight or volume. Ensure the nutritional profile aligns with the first few ingredients (e.g., if sugar or edible vegetable oil is listed first, the carbohydrate or fat values must logically reflect that high proportion).
    3. Mathematical Accuracy: Total Energy must align with standard conversion factors: Protein (~4 kcal/g), Carbohydrates (~4 kcal/g), and Fat (~9 kcal/g).
    4. Structural Integrity: Sub-components cannot exceed their parent category. Added Sugars must be ≤ Total Sugars. Total Sugars must be ≤ Carbohydrates. Saturated Fat + Trans Fat must be ≤ Total Fat. The sum of all macronutrients cannot exceed the serving size weight.
    5. Terminology & Additives: Scan the ingredients for generic/non-standard chemical names, blatant typos, or missing INS (International Numbering System) codes where specific food additives, colors, or preservatives are claimed.

    Based on your audit, determine if the data is compliant and valid, or if it contains regulatory, logical, or mathematical errors.

    You must reply with absolutely no conversational text, no explanations, and no markdown formatting (do not use ```json or backticks).

    Return ONLY a JSON object in the exact following format:
    {
      "is_data_valid": boolean,
      "detected_errors": ["List each specific FSSAI regulatory violation, mathematical discrepancy, or typo found strictly within the Nutrition Facts and Ingredients provided"]
    }"""

    user_content = f"Analyze this food label data: {json.dumps(targeted_data)}"

    try:
        # Generate content with the new client.models service
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                # This enforces JSON mode at the API level
                response_mime_type="application/json",
                # This tells Gemini exactly what the 'is_data_valid' and 'detected_errors' look like
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "is_data_valid": {"type": "BOOLEAN"},
                        "detected_errors": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        },
                    },
                    "required": ["is_data_valid", "detected_errors"],
                },
                temperature=0.0,
            ),
        )

        # In the 2026 SDK, 'parsed' is the cleanest way to get your dictionary
        if response.parsed:
            return response.parsed

        # Fallback if parsing fails
        return json.loads(response.text or "{}")

    except Exception as e:
        print(f"      [!] Gemini API Error: {e}")
        return {"error": "API_FAILED", "details": str(e)}


def process_all_files():
    for input_file in INPUT_FILES:
        if not os.path.exists(input_file):
            continue

        output_file = input_file.replace(".jsonl", "_evaluated_gemini.jsonl")
        print(f"\n>>> Processing with Gemini: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        with open(output_file, "w", encoding="utf-8") as out_f:
            for i, line in enumerate(lines):
                entry = json.loads(line)

                if entry.get("ground_truth_status") == "PENDING_REVIEW":
                    print(
                        f"    [{i + 1}/{len(lines)}] Product: {entry.get('product_id')}"
                    )

                    # Store the result
                    entry["ground_truth_status"] = analyze_with_gemini(entry)

                # Write back to file
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                # Free Tier Rate Limit: 10-15 RPM.
                # 6 seconds between requests is safe and reliable.
                time.sleep(5)

        print(f"[+] Results saved to: {output_file}")


if __name__ == "__main__":
    process_all_files()

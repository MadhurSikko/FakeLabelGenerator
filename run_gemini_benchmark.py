import json
import time

from google import genai
from google.genai import types
from pydantic import BaseModel

# 1. Configure the new Client
# Replace with your actual key from Google AI Studio
client = genai.Client(api_key="")


# 2. Define the exact JSON structure using Pydantic
class EvaluationResult(BaseModel):
    predicted_status: str
    reasoning: str
    flagged_ingredients: list[str]


def evaluate_with_gemini(product):
    system_instructions = (
        "You are a Senior Food Safety Scientist and FSSAI Regulatory Auditor. "
        "Your goal is to evaluate if the INGREDIENT FORMULATION of a product is permissible "
        "and considered 'Safe/OK' for its specific FSSAI Food Category.\n\n"
        "SCOPE OF EVALUATION:\n"
        "1. PERMITTED ADDITIVES: Check if preservatives, colors, and sweeteners (INS numbers) "
        "are approved for this specific Category (e.g., 'Carbonated Water' vs 'Dairy-based drinks').\n"
        "2. PROHIBITED COMBINATIONS: Identify if any ingredients are used in combinations prohibited by FSSAI.\n"
        "3. INGREDIENT INTEGRITY: Flag ingredients that are technically legal but considered 'Substandard' "
        "or 'Health-Negative' within the context of FSSAI's 'Eat Right India' initiatives.\n\n"
        "CRITICAL RULE: Ignore the presence or absence of statutory warnings (like 'Contains Caffeine'). "
        "Focus exclusively on whether the physical ingredients listed are ALLOWED to be in a product "
        "of this category according to the FSSAI Food Categorization System (FCS)."
    )

    user_prompt = f"""
    Evaluate the following product:
    Category: {product.get("fssai_category", "Unknown")}
    Ingredients: {product.get("raw_label_text", "None")}
    """

    try:
        # Call the model using the stable free tier
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"{system_instructions}\n\n{user_prompt}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvaluationResult,
                temperature=0.0,  # Zero temperature for maximum factual consistency
            ),
        )

        # The basedpyright fix: Ensure we actually got text back before parsing
        if not response.text:
            raise ValueError(
                "Gemini returned an empty response (possibly blocked by safety filters)."
            )

        # Parse the guaranteed JSON text returned by Gemini
        ai_output = json.loads(response.text)
        return ai_output

    except Exception as e:
        print(f"API Error for {product.get('product_name')}: {e}")
        return {
            "predicted_status": "Error",
            "reasoning": str(e),
            "flagged_ingredients": [],
        }


# 3. Read the input JSONL and remove duplicates
input_filename = "fssai_benchmark_dataset.jsonl"
output_filename = "benchmark_results_with_productName.jsonl"

# Load all products into a dictionary to automatically overwrite duplicates
unique_products = {}
try:
    with open(input_filename, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                prod = json.loads(line)
                # Using barcode as the key ensures each product only exists once
                unique_products[prod.get("product_id", "unknown")] = prod
except FileNotFoundError:
    print(f"Error: Could not find {input_filename}. Make sure the file exists.")
    exit(1)

# Convert the cleaned dictionary back into a list
product_list = list(unique_products.values())
total_products = len(product_list)

print(f"Starting Gemini Benchmark on {total_products} unique products...\n")

# 4. Run the evaluation loop
# We open in 'w' mode here to create a fresh results file, completely wiping any old test runs
with open(output_filename, "w", encoding="utf-8") as outfile:
    for index, product in enumerate(product_list, start=1):
        print(
            f"Evaluating [{index}/{total_products}]: {product.get('product_name', 'Unknown')}..."
        )

        # Send to Gemini
        ai_result = evaluate_with_gemini(product)

        # Merge Gemini's evaluation with your original data
        product["ai_predicted_status"] = ai_result.get("predicted_status", "Error")
        product["ai_reasoning"] = ai_result.get("reasoning", "No reasoning provided")
        product["ai_flagged_ingredients"] = ai_result.get("flagged_ingredients", [])

        # Write the updated JSON object as a new line in the results file
        json.dump(product, outfile, ensure_ascii=False)
        outfile.write("\n")

        # Brief pause to respect API rate limits
        time.sleep(5)

print("\nBenchmark complete! Check benchmark_results.jsonl for your clean data.")

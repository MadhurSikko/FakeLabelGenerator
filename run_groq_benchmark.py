import json
import time

from groq import Groq  # Switched from google.genai
from pydantic import BaseModel, Field

# 1. Configure the Groq Client
# Replace with your actual Groq API Key from console.groq.com
client = Groq(api_key="")


# 2. Define the exact JSON structure (Used for validation)
class EvaluationResult(BaseModel):
    predicted_status: str
    reasoning: str
    # If the AI sends null or nothing, this defaults to an empty list []
    flagged_ingredients: list[str] = Field(default_factory=list)


def evaluate_with_groq(product):
    system_instructions = (
        "You are a Senior Food Safety Scientist and FSSAI Regulatory Auditor. "
        "Your goal is to evaluate if the INGREDIENT FORMULATION of a product is permissible "
        "and considered 'Safe/OK' for its specific FSSAI Food Category.\n\n"
        "Output strictly as a JSON object with the following keys: "
        "'predicted_status', 'reasoning', and 'flagged_ingredients'."
    )

    user_prompt = f"""
    Evaluate the following product:
    Category: {product.get("fssai_category", "Unknown")}
    Ingredients: {product.get("raw_label_text", "None")}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        raw_content = chat_completion.choices[0].message.content

        if raw_content is None:
            raise ValueError("Groq returned None instead of a response.")

        ai_output_dict = json.loads(raw_content)

        # --- THE FIX: Handle nested wrappers (like 'evaluation' or 'result') ---
        # If the AI wrapped the response in a single key, dive into that key
        if len(ai_output_dict) == 1 and isinstance(
            list(ai_output_dict.values())[0], dict
        ):
            ai_output_dict = list(ai_output_dict.values())[0]

        # Final check: Ensure keys match what Pydantic expects
        # (This maps 'health_status' to 'predicted_status' if the AI got confused)
        if ai_output_dict.get("flagged_ingredients") is None:
            ai_output_dict["flagged_ingredients"] = []

        if (
            "health_status" in ai_output_dict
            and "predicted_status" not in ai_output_dict
        ):
            ai_output_dict["predicted_status"] = ai_output_dict.pop("health_status")

        validated_output = EvaluationResult(**ai_output_dict)
        return validated_output.model_dump()

    except Exception as e:
        print(f"API Error for {product.get('product_name')}: {e}")
        return {
            "predicted_status": "Error",
            "reasoning": f"Validation/API Error: {str(e)}",
            "flagged_ingredients": [],
        }


# 3. File Handling (Remains the same)
input_filename = "fssai_benchmark_dataset.jsonl"
output_filename = "benchmark_results_groq.jsonl"

unique_products = {}
try:
    with open(input_filename, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                prod = json.loads(line)
                unique_products[prod.get("product_id", "unknown")] = prod
except FileNotFoundError:
    print(f"Error: Could not find {input_filename}.")
    exit(1)

product_list = list(unique_products.values())
total_products = len(product_list)

print(f"Starting Groq Benchmark on {total_products} unique products...\n")

# 4. Evaluation Loop
with open(output_filename, "w", encoding="utf-8") as outfile:
    for index, product in enumerate(product_list, start=1):
        print(
            f"Evaluating [{index}/{total_products}]: {product.get('product_name', 'Unknown')}..."
        )

        ai_result = evaluate_with_groq(product)

        product["ai_predicted_status"] = ai_result.get("predicted_status", "Error")
        product["ai_reasoning"] = ai_result.get("reasoning", "No reasoning provided")
        product["ai_flagged_ingredients"] = ai_result.get("flagged_ingredients", [])

        json.dump(product, outfile, ensure_ascii=False)
        outfile.write("\n")

        # Groq is much faster; we can reduce wait time to 1 second
        time.sleep(1)

print(f"\nBenchmark complete! Check {output_filename} for your data.")

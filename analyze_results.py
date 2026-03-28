import json


def run_analysis(filename="benchmark_results_groq.jsonl"):
    total = 0
    correct = 0
    hallucinations = 0
    api_errors = 0

    # Categories for qualitative analysis
    hallucinated_products = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            total += 1

            status = data.get("ai_predicted_status", "Error")

            # Since all products in our dataset are Tier-1 Verified safe:
            # "Safe/OK" or "Compliant" = Correct (True Negative)
            # "Non-Compliant" = Hallucination (False Positive)

            if status in ["Safe/OK", "Compliant"]:
                correct += 1
            elif status == "Non-Compliant":
                hallucinations += 1
                hallucinated_products.append(
                    {
                        "name": data.get("product_name"),
                        "reason": data.get("ai_reasoning"),
                    }
                )
            else:
                api_errors += 1

    # Math
    processed = total - api_errors
    accuracy = (correct / processed * 100) if processed > 0 else 0
    fpr = (hallucinations / processed * 100) if processed > 0 else 0

    print("-" * 30)
    print("FSSAI BENCHMARK RESULTS")
    print("-" * 30)
    print(f"Total Products Tested: {total}")
    print(f"Successful API Calls:  {processed}")
    print(f"Failed API Calls:      {api_errors}")
    print("-" * 30)
    print(f"True Negatives (Correct): {correct}")
    print(f"False Positives (Hallucinations): {hallucinations}")
    print("-" * 30)
    print(f"ACCURACY: {accuracy:.2f}%")
    print(f"HALLUCINATION RATE (FPR): {fpr:.2f}%")
    print("-" * 30)

    if hallucinated_products:
        print("\nTOP HALLUCINATIONS TO ANALYZE:")
        for p in hallucinated_products:
            print(f"- {p['name']}: {p['reason'][:100]}...")


if __name__ == "__main__":
    run_analysis()

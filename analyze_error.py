import json
import os


def analyze_results():
    files = [
        "dataset_1_error_evaluated_gemini.jsonl",
        "dataset_2_error_evaluated_gemini.jsonl",
        "dataset_3_error_evaluated_gemini.jsonl",
    ]

    print(f"{'File Source':<40} | {'Total'} | {'Caught'} | {'Missed'} | {'Accuracy'}")
    print("-" * 85)

    grand_total = 0
    grand_caught = 0

    for file_name in files:
        if not os.path.exists(file_name):
            print(f"[-] Missing: {file_name}")
            continue

        total_count = 0
        caught_count = 0
        missed_count = 0
        api_failures = 0

        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                evaluation = data.get("ground_truth_status", {})

                # Skip if the API failed for this specific row
                if "error" in evaluation:
                    api_failures += 1
                    continue

                total_count += 1

                # Logic: Since these are ERROR datasets,
                # 'is_data_valid' SHOULD be False.
                if evaluation.get("is_data_valid") is False:
                    caught_count += 1
                else:
                    missed_count += 1

        # Calculate Accuracy %
        accuracy = (caught_count / total_count * 100) if total_count > 0 else 0

        grand_total += total_count
        grand_caught += caught_count

        print(
            f"{file_name:<40} | {total_count:>5} | {caught_count:>6} | {missed_count:>6} | {accuracy:>7.1f}%"
        )
        if api_failures > 0:
            print(f"    (Note: {api_failures} API failures skipped in this file)")

    # Final Totals
    print("-" * 85)
    total_accuracy = (grand_caught / grand_total * 100) if grand_total > 0 else 0
    print(
        f"{'OVERALL PERFORMANCE':<40} | {grand_total:>5} | {grand_caught:>6} | {'-':>6} | {total_accuracy:>7.1f}%"
    )


if __name__ == "__main__":
    analyze_results()

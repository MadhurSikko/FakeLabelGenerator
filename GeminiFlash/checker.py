import json
import os


def check_api_limits(files):
    grand_total = 0
    grand_success = 0
    grand_rate_limit = 0

    print("=== Batch API Limit Evaluation Report ===")

    for file_path in files:
        # Skip gracefully if a file is missing
        if not os.path.exists(file_path):
            print(f"\n[Warning] File not found: {file_path}")
            continue

        file_total = 0
        file_success = 0
        file_rate_limit = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                file_total += 1

                try:
                    record = json.loads(line)

                    # Safely navigate the JSON structure
                    # Default to an empty dict/string if keys are missing
                    status = record.get("ground_truth_status", {})

                    # Sometimes error might be None instead of a string, so we force it to string
                    error_msg = str(status.get("error") or "")

                    # Check for quota limit flags
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        file_rate_limit += 1
                    else:
                        file_success += 1

                except json.JSONDecodeError:
                    print(
                        f"  [Error] Skipping invalid JSON on line {line_num} in {file_path}"
                    )

        # Accumulate grand totals
        grand_total += file_total
        grand_success += file_success
        grand_rate_limit += file_rate_limit

        # Print individual file report
        print(f"\nFile: {file_path}")
        print(
            f"  -> Total: {file_total} | Success: {file_success} | 429 Errors: {file_rate_limit}"
        )

    # Print grand total report
    print("\n=========================================")
    print("              GRAND TOTALS               ")
    print("=========================================")
    print(f"Total Records Checked:     {grand_total}")
    print(f"Total Successful:          {grand_success}")
    print(f"Total Rate Limits (429):   {grand_rate_limit}")


if __name__ == "__main__":
    # The array of your specific benchmark files
    target_files = [
        "fssai_benchmark_dataset_1error_gemini_reviewed.jsonl",
        "fssai_benchmark_dataset_2errors_gemini_reviewed.jsonl",
        "fssai_benchmark_dataset_3errors_gemini_reviewed.jsonl",
        "fssai_benchmark_dataset_fictitious_compounds_gemini_compound_reviewed.jsonl",
        "fssai_benchmark_dataset_ins_errors_gemini_ins_reviewed.jsonl",
    ]

    check_api_limits(target_files)

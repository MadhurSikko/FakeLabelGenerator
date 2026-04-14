import csv
import json

files = [
    "fssai_benchmark_dataset_1error_openai_reviewed.jsonl",
    "fssai_benchmark_dataset_2errors_openai_reviewed.jsonl",
    "fssai_benchmark_dataset_3errors_openai_reviewed.jsonl",
    "fssai_benchmark_dataset_fictitious_compounds_gemini_compound_reviewed.jsonl",
    "fssai_benchmark_dataset_ins_errors_gemini_ins_reviewed.jsonl",
]

results = []

for f in files:
    total_labels = 0
    labels_flagged = 0
    total_injected = 0
    total_flagged = 0
    total_correct = 0

    with open(f, "r") as fp:
        for line in fp:
            data = json.loads(line)

            injected_list = []
            flagged_list = []

            # Extract true injected errors
            if "injected_errors" in data:
                injected_list = data["injected_errors"]
            elif "ins_changes" in data:
                injected_list = data["ins_changes"]
            elif (
                "fictitious_compound_injected" in data
                and data["fictitious_compound_injected"]
            ):
                if "fictitious_compound_changes" in data:
                    injected_list = [data["fictitious_compound_changes"]]

            # Extract errors flagged by the LLM
            gt = data.get("ground_truth_status", {})
            if "errors" in gt:
                flagged_list = gt["errors"]
            elif "flagged_codes" in gt:
                flagged_list = gt["flagged_codes"]
            elif "suspicious_compounds" in gt:
                flagged_list = gt["suspicious_compounds"]

            num_injected = len(injected_list)
            num_flagged = len(flagged_list)

            # Compute label-level counts
            if num_injected > 0:
                total_labels += 1
                if num_flagged > 0:
                    labels_flagged += 1

            total_injected += num_injected
            total_flagged += num_flagged

            correct = 0
            flagged_matched = set()
            for inj in injected_list:
                for i, flag in enumerate(flagged_list):
                    if i in flagged_matched:
                        continue

                    is_match = False

                    if "corrupted" in inj and "original" in flag:
                        if inj["corrupted"].lower() == flag["original"].lower():
                            is_match = True
                        elif (
                            abs(
                                inj.get("char_index", -999)
                                - flag.get("char_index", 999)
                            )
                            < 20
                        ):
                            is_match = True
                    elif "replacement_code" in inj and "ins_code" in flag:
                        if (
                            str(inj["replacement_code"]).lower()
                            in str(flag["ins_code"]).lower()
                        ):
                            is_match = True
                    elif "ins_code" in inj and "ins_code" in flag:
                        if (
                            str(inj["ins_code"]).lower()
                            in str(flag["ins_code"]).lower()
                        ):
                            is_match = True
                        elif "compound_name" in inj and "compound_name" in flag:
                            if (
                                inj["compound_name"].lower()
                                in flag["compound_name"].lower()
                            ):
                                is_match = True

                    if is_match:
                        correct += 1
                        flagged_matched.add(i)
                        break

            total_correct += correct

    # Calculate final metrics with percentages
    detection_rate = (labels_flagged / total_labels) * 100 if total_labels > 0 else 0
    error_recall = (total_correct / total_injected) * 100 if total_injected > 0 else 0
    identification_accuracy = (
        (total_correct / total_flagged) * 100 if total_flagged > 0 else 0
    )
    critical_omission_rate = 100 - detection_rate

    file_short_name = f.replace("fssai_benchmark_dataset_", "").replace(".jsonl", "")

    # Constrain appended dictionary to only the required evaluation metrics
    results.append(
        {
            "Dataset": file_short_name,
            "Detection Rate": f"{detection_rate:.2f}%",
            "Error Recall": f"{error_recall:.2f}%",
            "Identification Accuracy": f"{identification_accuracy:.2f}%",
            "Critical Omission Rate": f"{critical_omission_rate:.2f}%",
        }
    )

# Format and print the Markdown table
headers = list(results[0].keys())
header_row = "| " + " | ".join(headers) + " |"
separator_row = "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|"

print(header_row)
print(separator_row)

for row in results:
    row_str = "| " + " | ".join(str(row[h]) for h in headers) + " |"
    print(row_str)

# Write to CSV
with open("fssai_evaluation_metrics_simplified.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    writer.writerows(results)

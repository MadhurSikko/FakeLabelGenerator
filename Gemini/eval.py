import csv
import json
import os


def evaluate_spelling(file_path, test_name):
    tp_label, fn_label = 0, 0
    total_injected, total_found, total_caught = 0, 0, 0

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            injected = data.get("injected_errors", [])
            llm_review = data.get("ground_truth_status", {})
            llm_errors = llm_review.get("errors", [])

            # Label-Level Detection
            if len(injected) > 0:
                if len(llm_errors) > 0:
                    tp_label += 1
                else:
                    fn_label += 1

            total_injected += len(injected)
            total_found += len(llm_errors)

            # Entity-Level Match: Did the LLM catch the specific corrupted word?
            for inj in injected:
                corrupted = inj["corrupted"].lower()
                for err in llm_errors:
                    if err.get("original", "").lower() == corrupted:
                        total_caught += 1
                        break

    return {
        "Test Phase": test_name,
        "Detection Rate (%)": round(
            (tp_label / (tp_label + fn_label)) * 100 if (tp_label + fn_label) else 0, 2
        ),
        "Error Recall (%)": round(
            (total_caught / total_injected) * 100 if total_injected else 0, 2
        ),
        "Identification Accuracy (%)": round(
            (total_caught / total_found) * 100 if total_found else 0, 2
        ),
    }


def evaluate_fictitious(file_path, test_name):
    tp_label, fn_label = 0, 0
    total_injected, total_found, total_caught = 0, 0, 0

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            changes = data.get("fictitious_compound_changes", {})
            llm_review = data.get("ground_truth_status", {})
            found_compounds = llm_review.get("suspicious_compounds", [])

            if data.get("fictitious_compound_injected"):
                total_injected += 1
                total_found += len(found_compounds)

                # Label-Level Detection
                if llm_review.get("fictitious_compounds_found", 0) > 0:
                    tp_label += 1
                else:
                    fn_label += 1

                inj_name = changes.get("compound_name", "").lower()

                # Entity-Level Match
                for c in found_compounds:
                    if (
                        inj_name in c.get("compound_name", "").lower()
                        or c.get("compound_name", "").lower() in inj_name
                    ):
                        total_caught += 1
                        break

    return {
        "Test Phase": test_name,
        "Detection Rate (%)": round(
            (tp_label / (tp_label + fn_label)) * 100 if (tp_label + fn_label) else 0, 2
        ),
        "Error Recall (%)": round(
            (total_caught / total_injected) * 100 if total_injected else 0, 2
        ),
        "Identification Accuracy (%)": round(
            (total_caught / total_found) * 100 if total_found else 0, 2
        ),
    }


def evaluate_ins(file_path, test_name):
    tp_label, fn_label = 0, 0
    total_injected, total_found, total_caught, total_correctly_flagged = 0, 0, 0, 0

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            changes = data.get("ins_changes", [])
            llm_review = data.get("ground_truth_status", {})
            flagged = llm_review.get("flagged_codes", [])

            if len(changes) > 0:
                total_injected += len(changes)
                total_found += len(flagged)

                if llm_review.get("ins_errors_found", 0) > 0:
                    tp_label += 1
                else:
                    fn_label += 1

                # Recall Check: How many injected codes were caught?
                for ch in changes:
                    rep_code = str(ch.get("replacement_code")).lower()
                    for fl in flagged:
                        if (
                            rep_code in str(fl.get("ins_code")).lower()
                            or str(fl.get("ins_code")).lower() in rep_code
                        ):
                            total_caught += 1
                            break

                # Accuracy Check: How many flagged codes were actual true errors?
                for fl in flagged:
                    fl_code = str(fl.get("ins_code")).lower()
                    for ch in changes:
                        if (
                            fl_code in str(ch.get("replacement_code")).lower()
                            or str(ch.get("replacement_code")).lower() in fl_code
                        ):
                            total_correctly_flagged += 1
                            break

    return {
        "Test Phase": test_name,
        "Detection Rate (%)": round(
            (tp_label / (tp_label + fn_label)) * 100 if (tp_label + fn_label) else 0, 2
        ),
        "Error Recall (%)": round(
            (total_caught / total_injected) * 100 if total_injected else 0, 2
        ),
        "Identification Accuracy (%)": round(
            (total_correctly_flagged / total_found) * 100 if total_found else 0, 2
        ),
    }


if __name__ == "__main__":
    results = []

    r1 = evaluate_spelling(
        "fssai_benchmark_dataset_1error_gemini_reviewed.jsonl", "Spelling: 1 Error"
    )
    if r1:
        results.append(r1)
    r2 = evaluate_spelling(
        "fssai_benchmark_dataset_2errors_gemini_reviewed.jsonl", "Spelling: 2 Errors"
    )
    if r2:
        results.append(r2)
    r3 = evaluate_spelling(
        "fssai_benchmark_dataset_3errors_gemini_reviewed.jsonl", "Spelling: 3 Errors"
    )
    if r3:
        results.append(r3)
    r4 = evaluate_fictitious(
        "fssai_benchmark_dataset_fictitious_compounds_gemini_compound_reviewed.jsonl",
        "Fictitious Compounds",
    )
    if r4:
        results.append(r4)
    r5 = evaluate_ins(
        "fssai_benchmark_dataset_ins_errors_gemini_ins_reviewed.jsonl",
        "INS Codes Scrambling",
    )
    if r5:
        results.append(r5)

    if results:
        # 1. Write to CSV using Python's built-in csv library
        csv_file = "benchmark_metrics_results.csv"
        headers = results[0].keys()

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)

        print(f"Successfully saved results to {csv_file}\n")

        # 2. Print a clean table to the terminal
        print(
            f"{'Test Phase':<25} | {'Detection Rate':<15} | {'Error Recall':<15} | {'ID Accuracy':<15}"
        )
        print("-" * 79)
        for r in results:
            print(
                f"{r['Test Phase']:<25} | {r['Detection Rate (%)']:>13.2f} % | {r['Error Recall (%)']:>13.2f} % | {r['Identification Accuracy (%)']:>11.2f} %"
            )
    else:
        print("No data processed. Check if the .jsonl files are in the same folder.")

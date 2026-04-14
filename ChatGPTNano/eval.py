"""
FSSAI Benchmark Evaluation Metrics
====================================
Calculates the following metrics across all benchmark JSONL files:

  1. Label Level Detection Rate  – Share of products where the LLM correctly
                                   flagged that *at least one* error / anomaly
                                   exists (product-level TP / (TP + FN)).

  2. Error Recall                – Share of individually injected errors that
                                   the LLM identified (instance-level recall).

  3. Identification Rate         – Share of LLM-detected items that were NOT
                                   actually injected (hallucination / false
                                   positive rate at the instance level).

  4. Critical Omission Rate      – Share of injected errors that the LLM
                                   completely missed (False Negative Rate at
                                   the instance level; complement of Recall).

Dataset variants handled
------------------------
  * spelling_errors  (1-error / 2-errors / 3-errors)  – openai_reviewed
  * fictitious_compounds                               – gemini_compound_reviewed
  * ins_errors                                         – gemini_ins_reviewed

Ground-truth signals
--------------------
  spelling  : injected_errors[]           vs  ground_truth_status.errors[]
  fictitious: fictitious_compound_changes  vs  ground_truth_status.suspicious_compounds[]
  ins       : ins_changes[]               vs  ground_truth_status.flagged_codes[]
"""

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _norm(text: str) -> str:
    """Lowercase + strip for fuzzy string matching."""
    return text.lower().strip()


# ---------------------------------------------------------------------------
# Per-record extraction  →  (n_injected, n_detected, n_true_positive)
# ---------------------------------------------------------------------------


def extract_spelling(record: dict) -> tuple[int, int, int]:
    """
    Injected ground truth : record['injected_errors']          – list of dicts
    LLM detection         : record['ground_truth_status']['errors'] – list of dicts

    A detection is a True Positive when the LLM-detected 'original' word
    matches an injected 'corrupted' word (case-insensitive).
    """
    injected = record.get("injected_errors", [])
    detected = record.get("ground_truth_status", {}).get("errors", [])

    n_injected = len(injected)
    n_detected = len(detected)

    # Build a set of corrupted words from injections (what the LLM should see)
    injected_corrupted = {_norm(e["corrupted"]) for e in injected if "corrupted" in e}

    # Count how many detected originals match an injected corrupted word
    tp = sum(1 for d in detected if _norm(d.get("original", "")) in injected_corrupted)

    return n_injected, n_detected, tp


def extract_fictitious(record: dict) -> tuple[int, int, int]:
    """
    Injected ground truth : record['fictitious_compound_changes'] – dict (single compound)
                            record['fictitious_compound_injected'] – bool
    LLM detection         : record['ground_truth_status']['suspicious_compounds'] – list

    A detection is a TP when the LLM-detected compound_name overlaps
    with the injected compound_name (case-insensitive substring match).
    """
    injected_flag = record.get("fictitious_compound_injected", False)
    changes = record.get("fictitious_compound_changes", {})

    n_injected = 1 if injected_flag else 0

    suspicious = record.get("ground_truth_status", {}).get("suspicious_compounds", [])
    n_detected = len(suspicious)

    if not injected_flag or not changes:
        # No injection – any detection is a false positive
        return n_injected, n_detected, 0

    injected_name = _norm(changes.get("compound_name", ""))
    injected_ins = _norm(str(changes.get("ins_code", "")))

    tp = 0
    for s in suspicious:
        det_name = _norm(s.get("compound_name", ""))
        det_ins = _norm(str(s.get("ins_code", "")))
        # Match on compound name OR INS code
        if (injected_name and injected_name in det_name) or (
            injected_ins and injected_ins in det_ins
        ):
            tp = 1  # at least one suspicious item matched the injection
            break

    return n_injected, n_detected, tp


def extract_ins(record: dict) -> tuple[int, int, int]:
    """
    Injected ground truth : record['ins_changes']           – list of dicts
    LLM detection         : record['ground_truth_status']['flagged_codes'] – list of dicts

    A detection is a TP when a flagged ins_code matches a replacement_code
    from the injected changes (string match).
    """
    ins_found = record.get("ins_code_found", False)
    changes = record.get("ins_changes", [])

    n_injected = len(changes) if ins_found else 0
    flagged = record.get("ground_truth_status", {}).get("flagged_codes", [])
    n_detected = len(flagged)

    if not ins_found or not changes:
        return n_injected, n_detected, 0

    injected_codes = {str(c.get("replacement_code", "")).strip() for c in changes}
    tp = sum(1 for f in flagged if str(f.get("ins_code", "")).strip() in injected_codes)
    return n_injected, n_detected, tp


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "name": "Spelling Errors – 1 error  (OpenAI)",
        "file": "fssai_benchmark_dataset_1error_openai_reviewed.jsonl",
        "extractor": extract_spelling,
        "type": "spelling",
    },
    {
        "name": "Spelling Errors – 2 errors (OpenAI)",
        "file": "fssai_benchmark_dataset_2errors_openai_reviewed.jsonl",
        "extractor": extract_spelling,
        "type": "spelling",
    },
    {
        "name": "Spelling Errors – 3 errors (OpenAI)",
        "file": "fssai_benchmark_dataset_3errors_openai_reviewed.jsonl",
        "extractor": extract_spelling,
        "type": "spelling",
    },
    {
        "name": "Fictitious Compounds        (Gemini)",
        "file": "fssai_benchmark_dataset_fictitious_compounds_gemini_compound_reviewed.jsonl",
        "extractor": extract_fictitious,
        "type": "fictitious",
    },
    {
        "name": "INS Code Errors             (Gemini)",
        "file": "fssai_benchmark_dataset_ins_errors_gemini_ins_reviewed.jsonl",
        "extractor": extract_ins,
        "type": "ins",
    },
]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(records: list[dict], extractor) -> dict[str, Any]:
    """
    Returns a dict with all four metrics plus raw counts.

    Definitions
    -----------
    Let per record:
        I  = number of injected (true) errors
        D  = number of LLM-detected items
        TP = true positives (correct detections)
        FP = D - TP  (hallucinations / false positives)
        FN = I - TP  (missed / false negatives)

    Product-level (for Detection Rate):
        product is Positive  if I > 0
        product is TP_prod   if (I > 0) AND (TP > 0)
        product is FN_prod   if (I > 0) AND (TP == 0)

    Metrics:
        Label Level Detection Rate = TP_prod / (TP_prod + FN_prod)
                                   = TP_prod / total_positive_products

        Error Recall               = sum(TP) / sum(I)

        Identification Rate        = sum(FP) / sum(D)          [hallucination rate]
          (If sum(D) == 0 this is defined as 0.)

        Critical Omission Rate     = sum(FN) / sum(I)          [false negative rate]
          (Complement of Error Recall.)
    """
    total_injected = 0
    total_detected = 0
    total_tp = 0

    positive_products = 0  # products that truly have errors
    tp_products = 0  # positive products where LLM caught ≥1 error

    for rec in records:
        n_inj, n_det, tp = extractor(rec)

        total_injected += n_inj
        total_detected += n_det
        total_tp += tp

        if n_inj > 0:
            positive_products += 1
            if tp > 0:
                tp_products += 1

    total_fp = total_detected - total_tp
    total_fn = total_injected - total_tp

    detection_rate = tp_products / positive_products if positive_products > 0 else 0.0
    error_recall = total_tp / total_injected if total_injected > 0 else 0.0
    identification_rate = total_fp / total_detected if total_detected > 0 else 0.0
    critical_omission_rate = total_fn / total_injected if total_injected > 0 else 0.0

    return {
        "n_records": len(records),
        "positive_products": positive_products,
        "tp_products": tp_products,
        "total_injected": total_injected,
        "total_detected": total_detected,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "detection_rate": detection_rate,
        "error_recall": error_recall,
        "identification_rate": identification_rate,
        "critical_omission_rate": critical_omission_rate,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[dict]) -> None:
    SEP = "=" * 90
    SEP2 = "-" * 90

    print(SEP)
    print("  FSSAI LLM BENCHMARK  –  Evaluation Metrics Report")
    print(SEP)

    metric_labels = {
        "detection_rate": "Label Level Detection Rate  (product TP / positive products)",
        "error_recall": "Error Recall                (instance TP / total injected)  ",
        "identification_rate": "Identification Rate         (instance FP / total detected)  ",
        "critical_omission_rate": "Critical Omission Rate      (instance FN / total injected)  ",
    }

    # ── per-dataset table ─────────────────────────────────────────────────
    for res in results:
        m = res["metrics"]
        print()
        print(f"  Dataset : {res['name']}")
        print(f"  File    : {res['file']}")
        print(SEP2)
        print(f"  Records              : {m['n_records']}")
        print(f"  Products with errors : {m['positive_products']}")
        print(f"  Total injected       : {m['total_injected']}")
        print(f"  Total detected       : {m['total_detected']}")
        print(f"  True  Positives (TP) : {m['total_tp']}")
        print(f"  False Positives (FP) : {m['total_fp']}")
        print(f"  False Negatives (FN) : {m['total_fn']}")
        print(SEP2)
        for key, label in metric_labels.items():
            val = m[key]
            print(f"  {label} : {val:.4f}  ({val * 100:.2f}%)")
        print(SEP2)

    # ── aggregated summary across ALL datasets ────────────────────────────
    print()
    print(SEP)
    print("  AGGREGATED SUMMARY  (all datasets combined)")
    print(SEP)

    agg: dict[str, float] = {
        "n_records": float(sum(r["metrics"]["n_records"] for r in results)),
        "positive_products": float(
            sum(r["metrics"]["positive_products"] for r in results)
        ),
        "tp_products": float(sum(r["metrics"]["tp_products"] for r in results)),
        "total_injected": float(sum(r["metrics"]["total_injected"] for r in results)),
        "total_detected": float(sum(r["metrics"]["total_detected"] for r in results)),
        "total_tp": float(sum(r["metrics"]["total_tp"] for r in results)),
        "total_fp": float(sum(r["metrics"]["total_fp"] for r in results)),
        "total_fn": float(sum(r["metrics"]["total_fn"] for r in results)),
    }
    agg["detection_rate"] = (
        agg["tp_products"] / agg["positive_products"]
        if agg["positive_products"]
        else 0.0
    )
    agg["error_recall"] = (
        agg["total_tp"] / agg["total_injected"] if agg["total_injected"] else 0.0
    )
    agg["identification_rate"] = (
        agg["total_fp"] / agg["total_detected"] if agg["total_detected"] else 0.0
    )
    agg["critical_omission_rate"] = (
        agg["total_fn"] / agg["total_injected"] if agg["total_injected"] else 0.0
    )

    print(f"  Total records              : {agg['n_records']}")
    print(f"  Products with errors       : {agg['positive_products']}")
    print(f"  Total injected errors      : {agg['total_injected']}")
    print(f"  Total LLM detections       : {agg['total_detected']}")
    print(f"  True  Positives (TP)       : {agg['total_tp']}")
    print(f"  False Positives (FP)       : {agg['total_fp']}")
    print(f"  False Negatives (FN)       : {agg['total_fn']}")
    print(SEP)
    for key, label in metric_labels.items():
        val = agg[key]
        print(f"  {label} : {val:.4f}  ({val * 100:.2f}%)")
    print(SEP)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Resolve file paths relative to this script's directory, then fall back
    # to the uploads directory used in the Claude.ai sandbox.
    script_dir = Path(__file__).parent
    uploads_dir = Path("/mnt/user-data/uploads")

    results = []
    missing = []

    for ds in DATASETS:
        for base_dir in (script_dir, uploads_dir):
            path = base_dir / ds["file"]
            if path.exists():
                break
        else:
            missing.append(ds["file"])
            continue

        records = load_jsonl(str(path))
        metrics = compute_metrics(records, ds["extractor"])
        results.append({"name": ds["name"], "file": ds["file"], "metrics": metrics})
        print(f"[OK] Loaded {len(records):>3} records  ←  {ds['file']}")

    if missing:
        print()
        print("WARNING – the following files were NOT found and were skipped:")
        for f in missing:
            print(f"  • {f}")

    if not results:
        print(
            "No datasets loaded. Please place the JSONL files in the same directory"
            " as this script (or in /mnt/user-data/uploads)."
        )
        return

    print_report(results)


if __name__ == "__main__":
    main()

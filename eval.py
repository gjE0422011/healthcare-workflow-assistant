"""Evaluate the rule-based components against labelled synthetic notes.

Usage:
    python eval.py                      # table + error details
    python eval.py --json results.json  # also save metrics
    python eval.py --data other.jsonl   # use a different dataset

Each JSONL row: {"id", "text", "phi": [strings], "tasks": [sentences],
"red_flags": [category names]}.
"""
import argparse
import json

from core import extract_tasks, red_flag_screen, scrub_phi


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_notes.jsonl")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rows = load(args.data)

    c = {k: [0, 0, 0] for k in ("phi", "tasks", "flags")}
    leaks, gold_phi_total = 0, 0
    errors = []

    for row in rows:
        text = row["text"]

        # PHI: span match on exact strings, plus a leakage check on the output
        scrubbed, found = scrub_phi(text)
        det = {s for _, s in found}
        gold = set(row["phi"])
        c["phi"][0] += len(det & gold)
        c["phi"][1] += len(det - gold)
        c["phi"][2] += len(gold - det)
        for g in gold:
            gold_phi_total += 1
            if g in scrubbed:
                leaks += 1
        for kind, items in (("PHI missed", gold - det), ("PHI extra", det - gold)):
            for it in items:
                errors.append((row["id"], kind, it))

        # Tasks: exact sentence match
        det_t = set(extract_tasks(text))
        gold_t = set(row["tasks"])
        c["tasks"][0] += len(det_t & gold_t)
        c["tasks"][1] += len(det_t - gold_t)
        c["tasks"][2] += len(gold_t - det_t)
        for kind, items in (("Task missed", gold_t - det_t), ("Task extra", det_t - gold_t)):
            for it in items:
                errors.append((row["id"], kind, it))

        # Red flags: category level
        det_f = set(red_flag_screen(text))
        gold_f = set(row["red_flags"])
        c["flags"][0] += len(det_f & gold_f)
        c["flags"][1] += len(det_f - gold_f)
        c["flags"][2] += len(gold_f - det_f)
        for kind, items in (("Flag missed", gold_f - det_f), ("Flag false alarm", det_f - gold_f)):
            for it in items:
                errors.append((row["id"], kind, it))

    results = {k: prf(*v) for k, v in c.items()}
    results["phi"]["leak_rate"] = leaks / gold_phi_total if gold_phi_total else 0.0
    results["n_notes"] = len(rows)

    print(f"Notes: {len(rows)}\n")
    print(f"{'Component':<12}{'Prec':>7}{'Recall':>8}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
    for name, key in (("PHI scrub", "phi"), ("Tasks", "tasks"), ("Red flags", "flags")):
        m = results[key]
        print(f"{name:<12}{m['precision']:>7.2f}{m['recall']:>8.2f}{m['f1']:>7.2f}"
              f"{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}")
    print(f"\nPHI leak rate (gold PHI still in scrubbed text): {results['phi']['leak_rate']:.2f}")

    print("\nErrors:")
    for nid, kind, item in errors:
        print(f"  note {nid:>2} | {kind:<16} | {item}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to {args.json}")


if __name__ == "__main__":
    main()

"""Shared test-set building and metrics for the baselines.

Two real MVTec test sets are reported for every binary model:
  targeted : test/good + crack + scratch + contamination  (26 good, 71 defects) -- the main set
  all      : test/good + every defect folder              (26 good, 141 defects)
"""
import glob
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

TARGETED = ["crack", "scratch", "contamination"]
SELECTED_THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def build_test_rows(mvtec_root, which):
    test_root = os.path.join(mvtec_root, "test")
    rows = []
    for cls in sorted(os.listdir(test_root)):
        cls_dir = os.path.join(test_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        if which == "targeted" and cls != "good" and cls not in TARGETED:
            continue
        for p in sorted(glob.glob(os.path.join(cls_dir, "*.png"))):
            rows.append({"path": p, "label": 0 if cls == "good" else 1, "source": f"real_test_{cls}"})
    return rows


def at_threshold(y, prob, th):
    pred = (prob >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(float(th), 3), "accuracy": round(accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4),
            "f1": round(f1_score(y, pred, zero_division=0), 4),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def report(y, prob, sources, default_threshold=0.5):
    """AUROC, metrics at the default threshold, the swept best-F1 threshold (chosen on this
    test set, so optimistic), selected thresholds, per-class recall, and the F1 of flagging
    everything as a defect (the floor any F1 must beat on an imbalanced test set)."""
    y, prob, sources = np.asarray(y), np.asarray(prob), np.asarray(sources)
    sweep = [at_threshold(y, prob, th) for th in np.arange(0.01, 1.00, 0.01)]
    best = max(sweep, key=lambda m: m["f1"])
    per_class = {}
    for s in sorted(set(sources)):
        if s.endswith("_good"):
            continue
        sel = sources == s
        per_class[s.replace("real_test_", "")] = {
            "n": int(sel.sum()),
            f"recall@{default_threshold}": round(float((prob[sel] >= default_threshold).mean()), 3),
            f"recall@{best['threshold']}": round(float((prob[sel] >= best["threshold"]).mean()), 3)}
    return {
        "n_good": int((y == 0).sum()), "n_defect": int((y == 1).sum()),
        "auroc": round(roc_auc_score(y, prob), 4),
        "flag_all_f1": round(f1_score(y, np.ones_like(y)), 4),
        "default": at_threshold(y, prob, default_threshold),
        "best_f1_threshold_on_test_OPTIMISTIC": best,
        "selected_thresholds": [at_threshold(y, prob, t) for t in SELECTED_THRESHOLDS],
        "per_class_recall": per_class,
    }


def print_report(name, r):
    d, b = r["default"], r["best_f1_threshold_on_test_OPTIMISTIC"]
    print(f"\n===== {name}: {r['n_good']} good / {r['n_defect']} defect =====")
    print(f"AUROC {r['auroc']:.3f} | flag-all F1 {r['flag_all_f1']:.3f}")
    print(f"default thr {d['threshold']}: P {d['precision']:.3f} R {d['recall']:.3f} F1 {d['f1']:.3f} "
          f"(TP {d['tp']} FN {d['fn']} FP {d['fp']} TN {d['tn']})")
    print(f"best-F1 thr {b['threshold']} (chosen on test): P {b['precision']:.3f} R {b['recall']:.3f} "
          f"F1 {b['f1']:.3f} (TP {b['tp']} FN {b['fn']} FP {b['fp']} TN {b['tn']})")
    for c, v in r["per_class_recall"].items():
        print(f"  {c}: {v}")


def save(out_dir, name, obj):
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, indent=2)

"""Test whether a trained baseline-3 model detects defects or just "generated-looking" images.

Scores the real MVTec test images twice with a trained model: as they are, and after a
Stable Diffusion VAE round trip (which changes nothing about the pill, only adds the
generation fingerprint). If P(defect) of *good* test images jumps after the round trip,
the model learned the fingerprint, not the defects.

    python diagnose_shortcut.py --mvtec D:\\...\\data\\pill --runs D:\\...\\runs\\b3_v1 D:\\...\\runs\\b3_v2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from baseline_3_binary import EVAL_TF
from eval_utils import build_test_rows
from vae_roundtrip import load_vae, roundtrip


@torch.no_grad()
def probs(model, device, images):
    x = torch.stack([EVAL_TF(im.convert("RGB")) for im in images]).to(device)
    return torch.softmax(model(x), 1)[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mvtec", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="baseline-3 output folders containing best.pt")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = build_test_rows(a.mvtec, "targeted")
    originals = [Image.open(r["path"]).convert("RGB") for r in rows]
    vae, dtype = load_vae(device)
    rt = []
    for i in range(0, len(originals), 8):
        rt += roundtrip(vae, dtype, device, originals[i:i + 8])
    del vae
    y = np.array([r["label"] for r in rows])

    out = {}
    for run in a.runs:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location=device))
        model = model.to(device).eval()
        p0 = np.concatenate([probs(model, device, originals[i:i + 16]) for i in range(0, len(rows), 16)])
        p1 = np.concatenate([probs(model, device, rt[i:i + 16]) for i in range(0, len(rows), 16)])
        res = {}
        for name, lab in (("good", 0), ("defect", 1)):
            s = y == lab
            res[name] = {"n": int(s.sum()),
                         "mean_p_defect_original": round(float(p0[s].mean()), 3),
                         "mean_p_defect_vae": round(float(p1[s].mean()), 3),
                         "share_flagged_original": round(float((p0[s] >= 0.5).mean()), 3),
                         "share_flagged_vae": round(float((p1[s] >= 0.5).mean()), 3)}
        out[os.path.basename(os.path.normpath(run))] = res
        print(f"\n{run}")
        for name, r in res.items():
            print(f"  real test {name:6s} (n={r['n']}): mean P(defect) {r['mean_p_defect_original']:.3f} -> "
                  f"{r['mean_p_defect_vae']:.3f} after VAE round trip; flagged "
                  f"{r['share_flagged_original']:.0%} -> {r['share_flagged_vae']:.0%}")
    with open(os.path.join(os.path.dirname(os.path.normpath(a.runs[0])), "shortcut_diagnosis.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

"""Baseline 3: binary ResNet18 (good vs defect) trained on synthetic defects, tested on real MVTec.

Versions, each with the original settings:
  v1                : real good + synthetic defects; 30 epochs, lr 1e-4, light colour jitter
  v2                : + synthetic good (blocks the "real = good, synthetic = defect" shortcut);
                      10 epochs, lr 5e-5
  v3_aug            : v2 + stronger augmentation (crop 0.82, rotation 15, stronger jitter,
                      Gaussian blur p=0.35, grayscale p=0.08, Gaussian noise std 0.015)
  v3_aug_nosyngood  : v3_aug without synthetic good (ablation)
  v4_vae            : v2, but every good training image (real + synthetic) is passed through the
                      SD VAE first (vae_roundtrip.py), so good and defect images share the
                      generation fingerprint and the model cannot use it as a shortcut
  v4_vae_aug        : v4_vae with the v3 augmentation

Common: 2-class softmax + cross entropy, WeightedRandomSampler, 20% stratified validation split,
AdamW + cosine schedule, best epoch by validation F1.

Evaluation is on two real test sets (see eval_utils): "targeted" (26 good, 71 crack/scratch/
contamination; the main set) and "all" (26 good, 141 defects).

Usage:
  python src/baseline_3_binary.py --version v3_aug --mvtec data/pill --syn work/pill_defects --out runs/b3_v3_aug
"""
import argparse
import csv
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from eval_utils import build_test_rows, print_report, report, save

DEFECT_CLASSES = ["crack", "scratch", "contamination", "contamination_raised"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

VERSIONS = {
    "v1": dict(syn_good=False, epochs=30, lr=1e-4, aug="v1"),
    "v2": dict(syn_good=True, epochs=10, lr=5e-5, aug="v2"),
    "v3_aug": dict(syn_good=True, epochs=10, lr=5e-5, aug="v3"),
    "v3_aug_nosyngood": dict(syn_good=False, epochs=10, lr=5e-5, aug="v3"),
    "v4_vae": dict(syn_good=True, epochs=10, lr=5e-5, aug="v2", vae_good=True),
    "v4_vae_aug": dict(syn_good=True, epochs=10, lr=5e-5, aug="v3", vae_good=True),
}


class AddGaussianNoise:
    """Picklable replacement for the original transforms.Lambda (needed for num_workers > 0)."""

    def __init__(self, std):
        self.std = std

    def __call__(self, x):
        return torch.clamp(x + torch.randn_like(x) * self.std, 0, 1)


def train_transform(kind):
    if kind == "v1":
        jitter, extra, scale, rot = transforms.ColorJitter(0.15, 0.15, 0.08), [], 0.85, 10
    elif kind == "v2":
        jitter, extra, scale, rot = transforms.ColorJitter(0.18, 0.18, 0.10, 0.02), [], 0.85, 10
    else:
        jitter, scale, rot = transforms.ColorJitter(0.22, 0.22, 0.12, 0.02), 0.82, 15
        extra = [transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))], p=0.35),
                 transforms.RandomGrayscale(p=0.08)]
    t = [transforms.Resize((256, 256)), transforms.RandomResizedCrop(224, scale=(scale, 1.0)),
         transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(rot), jitter, *extra,
         transforms.ToTensor()]
    if kind == "v3":
        t.append(AddGaussianNoise(0.015))
    t.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(t)


EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class BinaryPillDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.transform(Image.open(r["path"]).convert("RGB")), int(r["label"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=list(VERSIONS), required=True)
    ap.add_argument("--mvtec", required=True)
    ap.add_argument("--syn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="override the version default")
    ap.add_argument("--no-pretrained", action="store_true", help="offline smoke tests only")
    args = ap.parse_args()
    V = VERSIONS[args.version]
    epochs = args.epochs or V["epochs"]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    if V.get("vae_good"):
        vae_dir = os.path.join(args.syn, "good_vae", "images")
        real = sorted(glob.glob(os.path.join(vae_dir, "real_*.png")))
        syn = sorted(glob.glob(os.path.join(vae_dir, "syn_*.png")))
        if not real or not syn:
            raise SystemExit(f"{vae_dir} needs real_*.png and syn_*.png: run vae_roundtrip.py first (see README)")
        rows = [{"path": p, "label": 0, "source": "real_good_vae"} for p in real]
        rows += [{"path": p, "label": 0, "source": "synthetic_good_vae"} for p in syn]
        counts = {"real_good_vae": len(real), "synthetic_good_vae": len(syn)}
    else:
        rows = [{"path": p, "label": 0, "source": "real_good"}
                for p in sorted(glob.glob(os.path.join(args.mvtec, "train", "good", "*.png")))]
        counts = {"real_good": len(rows)}
    if V["syn_good"] and not V.get("vae_good"):
        g = sorted(glob.glob(os.path.join(args.syn, "good", "images", "*.png")))
        rows += [{"path": p, "label": 0, "source": "synthetic_good"} for p in g]
        counts["synthetic_good"] = len(g)
    for cls in DEFECT_CLASSES:
        d = sorted(glob.glob(os.path.join(args.syn, cls, "images", "*.png")))
        rows += [{"path": p, "label": 1, "source": f"synthetic_{cls}"} for p in d]
        counts[cls] = len(d)
    print(f"Baseline 3 {args.version} training data:", counts)

    labels = [r["label"] for r in rows]
    tr, va = train_test_split(rows, test_size=0.20, random_state=args.seed, stratify=labels)
    y_tr = np.array([r["label"] for r in tr])
    cc = np.bincount(y_tr, minlength=2)
    sampler = WeightedRandomSampler(torch.DoubleTensor((1.0 / np.maximum(cc, 1))[y_tr]), len(y_tr), replacement=True)
    tl = DataLoader(BinaryPillDataset(tr, train_transform(V["aug"])), batch_size=args.batch, sampler=sampler,
                    num_workers=args.workers, pin_memory=True)
    vl = DataLoader(BinaryPillDataset(va, EVAL_TF), batch_size=args.batch, num_workers=args.workers, pin_memory=True)

    model = models.resnet18(weights=None if args.no_pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=V["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def run(loader, train):
        model.train() if train else model.eval()
        tot, ys, ps, pr = 0.0, [], [], []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = crit(logits, y)
                if train:
                    loss.backward()
                    opt.step()
            tot += loss.item() * x.size(0)
            ys.extend(y.cpu().numpy()); ps.extend(logits.argmax(1).cpu().numpy())
            pr.extend(torch.softmax(logits, 1)[:, 1].detach().cpu().numpy())
        p, r, f1, _ = precision_recall_fscore_support(ys, ps, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(ys, pr)
        except ValueError:
            auc = float("nan")
        return tot / len(loader.dataset), accuracy_score(ys, ps), f1, auc

    best, history = -1.0, []
    for ep in range(1, epochs + 1):
        trl, tra, trf, tra_auc = run(tl, True)
        val, vaa, vaf, va_auc = run(vl, False)
        sched.step()
        history.append({"epoch": ep, "train_loss": trl, "val_loss": val, "val_f1": vaf, "val_auc": va_auc})
        print(f"Epoch {ep:02d}/{epochs} | Train loss {trl:.4f} acc {tra:.4f} f1 {trf:.4f} | "
              f"Val loss {val:.4f} acc {vaa:.4f} f1 {vaf:.4f} auc {va_auc:.4f}")
        if vaf > best:
            best = vaf
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
    print("Best val F1:", best, "(validation = synthetic + real good, not the real test set)")

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device))
    model.eval()
    results = {"version": args.version, "settings": {**V, "epochs": epochs}, "train_data": counts, "history": history}
    for which in ["targeted", "all"]:
        test_rows = build_test_rows(args.mvtec, which)
        loader = DataLoader(BinaryPillDataset(test_rows, EVAL_TF), batch_size=args.batch, num_workers=args.workers)
        probs = []
        with torch.no_grad():
            for x, _ in loader:
                probs.extend(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())
        y = [r["label"] for r in test_rows]
        src = [r["source"] for r in test_rows]
        r = report(y, probs, src)
        results[which] = r
        print_report(f"Baseline 3 {args.version} [{which}]", r)
        with open(os.path.join(args.out, f"predictions_{which}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "source", "true_label", "prob_defect"])
            for row, p in zip(test_rows, probs):
                w.writerow([row["path"], row["source"], row["label"], round(float(p), 5)])
    save(args.out, "eval.json", results)


if __name__ == "__main__":
    main()

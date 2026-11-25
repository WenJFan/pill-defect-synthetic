"""Baseline 2: synthetic-only 4-class ResNet18 (good / crack / scratch / contamination).

Original script: contamination_raised is merged into contamination; 70/15/15 random split;
weighted sampler; 30 epochs, lr 1e-4; best epoch by validation macro-F1.

This only measures whether the generated classes are separable from each other. A random
split puts variants of the same 267 MVTec base images in train and test, so the score is
optimistic. --group-split keeps every base image (from manifest.csv) in one split only.

Usage:
  python src/baseline_2_synthetic_multiclass.py --syn work/pill_defects --out runs/b2 [--group-split]
"""
import argparse
import csv
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from eval_utils import save

CLASS_NAMES = ["good", "crack", "scratch", "contamination"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
LABEL_MAP = {"good": "good", "crack": "crack", "scratch": "scratch",
             "contamination": "contamination", "contamination_raised": "contamination"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class PillDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df, self.root, self.transform = df.reset_index(drop=True), root, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.root, row["path"])).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[row["label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", required=True, help="generation output folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-split", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="offline smoke tests only")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    base_of = {}
    mpath = os.path.join(args.syn, "manifest.csv")
    if os.path.exists(mpath):
        for r in csv.DictReader(open(mpath)):
            base_of[(r["class"], r["file"])] = r["base_image"]
    rows = []
    for folder, label in LABEL_MAP.items():
        d = os.path.join(args.syn, folder, "images")
        if not os.path.exists(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".png"):
                rows.append({"path": os.path.join(folder, "images", f), "label": label,
                             "group": base_of.get((folder, f), f"{folder}/{f}")})
    df = pd.DataFrame(rows)
    print(df["label"].value_counts())

    if args.group_split:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.seed)
        tr_idx, tmp_idx = next(gss.split(df, groups=df["group"]))
        train_df, temp_df = df.iloc[tr_idx], df.iloc[tmp_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=args.seed)
        va_idx, te_idx = next(gss2.split(temp_df, groups=temp_df["group"]))
        val_df, test_df = temp_df.iloc[va_idx], temp_df.iloc[te_idx]
    else:
        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=args.seed, stratify=df["label"])
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed, stratify=temp_df["label"])
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    y_tr = train_df["label"].map(CLASS_TO_IDX).values
    counts = np.bincount(y_tr, minlength=len(CLASS_NAMES))
    sampler = WeightedRandomSampler(torch.DoubleTensor((1.0 / np.maximum(counts, 1))[y_tr]), len(y_tr), replacement=True)
    tl = DataLoader(PillDataset(train_df, args.syn, train_tf), batch_size=args.batch, sampler=sampler, num_workers=args.workers)
    vl = DataLoader(PillDataset(val_df, args.syn, eval_tf), batch_size=args.batch, num_workers=args.workers)
    te = DataLoader(PillDataset(test_df, args.syn, eval_tf), batch_size=args.batch, num_workers=args.workers)

    model = models.resnet18(weights=None if args.no_pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def run(loader, train):
        model.train() if train else model.eval()
        tot, ys, ps = 0.0, [], []
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
        _, _, f1, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
        return tot / len(loader.dataset), accuracy_score(ys, ps), f1, ys, ps

    best = -1
    for ep in range(1, args.epochs + 1):
        trl, tra, trf, _, _ = run(tl, True)
        val, vaa, vaf, _, _ = run(vl, False)
        sched.step()
        print(f"Epoch {ep:02d}/{args.epochs} | Train loss {trl:.4f} acc {tra:.4f} macroF1 {trf:.4f} | "
              f"Val loss {val:.4f} acc {vaa:.4f} macroF1 {vaf:.4f}")
        if vaf > best:
            best = vaf
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=device))
    _, acc, f1, ys, ps = run(te, False)
    print(f"\n===== Baseline 2: synthetic-only multiclass ({'group' if args.group_split else 'random'} split) =====")
    print("Accuracy:", acc, "Macro F1:", f1)
    print(confusion_matrix(ys, ps))
    print(classification_report(ys, ps, target_names=CLASS_NAMES, zero_division=0))
    save(args.out, "eval.json", {"split": "group" if args.group_split else "random", "accuracy": round(acc, 4),
                                 "macro_f1": round(f1, 4), "confusion_matrix": confusion_matrix(ys, ps).tolist()})


if __name__ == "__main__":
    main()

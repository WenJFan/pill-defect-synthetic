"""Baseline 1: convolutional autoencoder trained on real good pills only (original script).

Anomaly score = mean reconstruction error; decision threshold = 95th percentile of the
train-good errors (chosen without looking at the test set).

Usage:
  python src/baseline_1_autoencoder.py --mvtec data/pill --out runs/b1_autoencoder
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval_utils import at_threshold, build_test_rows, print_report, report, save


class ImagePathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB"))


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def errors(model, loader, device):
    out = []
    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            out.extend(torch.mean((model(x) - x) ** 2, dim=[1, 2, 3]).cpu().numpy())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    tf = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])

    train_good = sorted(glob.glob(os.path.join(args.mvtec, "train", "good", "*.png")))
    loader = DataLoader(ImagePathDataset(train_good, tf), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers)
    model = ConvAutoEncoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x in loader:
            x = x.to(device)
            opt.zero_grad()
            loss = crit(model(x), x)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        print(f"Epoch {ep:02d}/{args.epochs} | Train recon loss: {tot / len(train_good):.6f}")
    torch.save(model.state_dict(), os.path.join(args.out, "autoencoder.pt"))

    eval_loader = DataLoader(ImagePathDataset(train_good, tf), batch_size=args.batch, num_workers=args.workers)
    threshold = float(np.percentile(errors(model, eval_loader, device), 95))
    print("Threshold from train good 95th percentile:", threshold)

    results = {"train_threshold_p95": threshold}
    for which in ["targeted", "all"]:
        rows = build_test_rows(args.mvtec, which)
        err = errors(model, DataLoader(ImagePathDataset([r["path"] for r in rows], tf),
                                       batch_size=args.batch, num_workers=args.workers), device)
        y = [r["label"] for r in rows]
        # error scale is arbitrary: scale by the train threshold so 1.0 = the p95 cut-off
        r = report(y, err / threshold / 2, [r["source"] for r in rows], default_threshold=0.5)
        r["at_train_p95_threshold"] = at_threshold(np.array(y), err, threshold)
        results[which] = r
        print_report(f"Baseline 1 autoencoder [{which}]", r)
        print("  at train-p95 threshold:", r["at_train_p95_threshold"])
    save(args.out, "eval.json", results)


if __name__ == "__main__":
    main()

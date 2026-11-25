"""Quality control for generated images (manual review + blacklist).

Folder layout: <syn>/<class>/images/*.png and <syn>/<class>/masks/*.png

  auto    : move defect images whose masked region barely differs from the base image
            (failed inpaints); needs manifest.csv and --mvtec
  sheets  : numbered contact sheets per class (mask outlined in red) for a fast manual review
  apply   : move every file named in blacklist.txt to <syn>/rejected/<class>/

blacklist.txt: one file name per line, e.g.  crack_012.png   (# starts a comment)

Usage:
  python src/clean.py auto   --syn work/pill_defects --mvtec data/pill --min-change 6
  python src/clean.py sheets --syn work/pill_defects
  python src/clean.py apply  --syn work/pill_defects --blacklist work/blacklist.txt
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

DEFECTS = ["crack", "scratch", "contamination", "contamination_raised"]


def class_of(name):
    return name.rsplit("_", 1)[0]


def reject(syn, cls, name):
    moved = False
    for sub in ["images", "masks"]:
        src = syn / cls / sub / name
        if src.exists():
            dst = syn / "rejected" / cls / sub
            dst.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst / name))
            moved = moved or sub == "images"
    return moved


def auto(syn, mvtec, min_change):
    moved, checked = 0, 0
    for row in csv.DictReader(open(syn / "manifest.csv")):
        cls = row["class"]
        if cls not in DEFECTS:
            continue
        img_p, mask_p = syn / cls / "images" / row["file"], syn / cls / "masks" / row["file"]
        base_p = Path(mvtec) / "train" / "good" / row["base_image"]
        if not (img_p.exists() and mask_p.exists() and base_p.exists()):
            continue
        img = cv2.imread(str(img_p)).astype(int)
        base = cv2.resize(cv2.imread(str(base_p)), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA).astype(int)
        m = cv2.imread(str(mask_p), 0) > 127
        checked += 1
        if not m.any() or np.abs(img - base)[m].mean() < min_change:
            moved += reject(syn, cls, row["file"])
    print(f"checked {checked}, auto-rejected {moved} near-unchanged defect images (mean change < {min_change})")


def sheets(syn, per_row=10, per_sheet=100, tile=150):
    out = syn / "sheets"
    out.mkdir(exist_ok=True)
    for cls_dir in sorted(p for p in syn.iterdir() if (p / "images").is_dir()):
        files = sorted((cls_dir / "images").glob("*.png"))
        for k in range(0, len(files), per_sheet):
            tiles = []
            for f in files[k:k + per_sheet]:
                im = cv2.resize(cv2.imread(str(f)), (tile, tile))
                mp = cls_dir / "masks" / f.name
                m = cv2.imread(str(mp), 0) if mp.exists() else None
                if m is not None:
                    m = cv2.resize(m, (tile, tile), interpolation=cv2.INTER_NEAREST)
                    cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(im, cnts, -1, (0, 0, 255), 1)
                cv2.putText(im, f.stem.split("_")[-1], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                tiles.append(im)
            while len(tiles) % per_row:
                tiles.append(np.zeros((tile, tile, 3), np.uint8))
            grid = np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])
            cv2.imwrite(str(out / f"{cls_dir.name}_{k // per_sheet + 1:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{cls_dir.name}: {len(files)} images")
    print(f"sheets in {out}  (tile number = file index, e.g. 12 on a crack sheet -> crack_012.png)")


def apply(syn, blacklist):
    moved = 0
    for line in open(blacklist):
        name = line.split("#")[0].strip()
        if name:
            moved += reject(syn, class_of(name), name)
    print(f"moved {moved} blacklisted images to {syn / 'rejected'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["auto", "sheets", "apply"])
    ap.add_argument("--syn", required=True)
    ap.add_argument("--mvtec", default=None)
    ap.add_argument("--min-change", type=float, default=6.0)
    ap.add_argument("--blacklist", default=None)
    a = ap.parse_args()
    syn = Path(a.syn)
    if a.cmd == "auto":
        auto(syn, a.mvtec, a.min_change)
    elif a.cmd == "sheets":
        sheets(syn)
    else:
        apply(syn, a.blacklist)


if __name__ == "__main__":
    main()

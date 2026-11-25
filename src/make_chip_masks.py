"""Build chip masks for the PillQC chip images (the repo ships images only).

A chip is a bite out of the pill's outline, so the defect region is
(convex hull of the pill) minus (the pill itself).

Usage:
  python src/make_chip_masks.py --pillqc data/pillQC --out work/pillqc_chip
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import list_images, pill_mask

CAPTION = "small chipped edge, recessed grey broken area, pill"


def chip_mask(rgb):
    pill = pill_mask(rgb, channel="red")
    cnts, _ = cv2.findContours(pill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    hull_m = np.zeros_like(pill)
    cv2.drawContours(hull_m, [hull], -1, 255, cv2.FILLED)
    diff = cv2.bitwise_and(hull_m, cv2.bitwise_not(pill))
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(diff)
    keep = np.zeros_like(diff)
    min_area = 0.0015 * int((pill > 0).sum())
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    # widen (about 8% of the pill diameter) so the mask also covers the broken face next to the gap
    diam = 2 * np.sqrt((pill > 0).sum() / np.pi)
    k = max(5, int(diam * 0.08)) | 1
    return cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pillqc", required=True, help="root of the cloned pillQC repo")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for p in list_images(Path(args.pillqc) / "images" / "chip"):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        m = chip_mask(rgb)
        if m.sum() == 0:
            skipped.append(p.name)
            continue
        cv2.imwrite(str(out / "images" / p.name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        mask_name = p.stem + "_mask.png"
        cv2.imwrite(str(out / "masks" / mask_name), m)
        rows.append({"image": f"images/{p.name}", "mask": f"masks/{mask_name}", "caption": CAPTION})

    with open(out / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # preview: image with mask outline, for a quick visual check
    tiles = []
    for r in rows[:48]:
        im = cv2.imread(str(out / r["image"]))
        m = cv2.imread(str(out / r["mask"]), 0)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, (0, 0, 255), 2)
        tiles.append(cv2.resize(im, (160, 160)))
    if tiles:
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
        cv2.imwrite(str(out / "preview.jpg"), grid)
    print(f"masks written: {len(rows)}; no chip found (skipped): {len(skipped)} {skipped}")


if __name__ == "__main__":
    main()

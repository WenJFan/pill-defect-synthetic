"""Rank generated defect images by simple quality measures so only the suspicious
ones need a human look (instead of reviewing all ~1,800 by eye).

For every defect image, compared with the good image it was made from (manifest.csv):
  vis        defect visibility vs the original pixels: the larger of the mean lightness change and the
             dark detail added inside the mask (Lab L, global lighting shift removed; low = invisible)
  color      colour inside the mask vs a ring around it, Lab a*/b* (high = unnatural colour blob)
  shape_iou  pill outline before vs after (low = pill deformed)
  mask_out   share of the mask that lies outside the pill (high = defect off the pill)
  spill      change outside the mask, after removing the global lighting shift (high = artefacts elsewhere)

"faint" flags the least visible --faint share of each class (default 15%); the other
flags use fixed thresholds. Writes:
  <out>/scores.csv                      all measures, flags, score
  <out>/suspects_<class>_NN.jpg         review sheets: full image + 2x zoom on the defect, with flags
  <out>/blacklist_suggested.txt         every flagged file (edit, then use with clean.py apply)

    python qc_rank.py --syn D:\\Work_Study\\pill_work\\pill_defects --mvtec D:\\Work_Study\\pill_work\\data\\pill
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from common import pill_mask

DEFECTS = ["crack", "scratch", "contamination", "contamination_raised"]
# "faint" is relative: the least visible --faint share of each class (visibility scales differ a
# lot between thin scratches and large smudges). The others are absolute.
MAX_COLOR = 14.0
MIN_SHAPE_IOU = 0.965
MAX_MASK_OUT = 0.50        # edge cracks legitimately overlap the background; flag only gross misses


def measures(img_bgr, base_bgr, mask):
    m = mask > 127
    if not m.any():
        return None
    pill_b = pill_mask(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)) > 0
    pill_o = pill_mask(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)) > 0
    inter, union = (pill_b & pill_o).sum(), (pill_b | pill_o).sum()
    # Judge only the part of the mask on the pill: edge cracks overlap the dark background,
    # which would otherwise look like a strong "defect".
    pm = m & pill_b
    if pm.sum() < 20:
        pm = m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    dil = cv2.dilate(m.astype(np.uint8), k) > 0
    ring = dil & ~m & pill_b
    if ring.sum() < 20:
        ring = pill_b & ~m
    far = pill_b & ~(cv2.dilate(dil.astype(np.uint8), k) > 0)
    lab_i = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(float)
    lab_b = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB).astype(float)
    L_i, L_b = lab_i[..., 0] * 100 / 255, lab_b[..., 0] * 100 / 255
    # remove the global lighting change applied after inpainting
    shift = float(np.median((L_i - L_b)[far])) if far.sum() > 100 else 0.0
    d = L_i - L_b - shift
    mean_change = abs(float(d[pm].mean()))
    # dark detail added inside the mask (thin scratches/cracks), relative to the same pixels before
    dark_i = np.median(L_i[ring]) - np.percentile(L_i[pm], 10)
    dark_b = np.median(L_b[ring]) - np.percentile(L_b[pm], 10)
    ab_in, ab_ring = lab_i[..., 1:][pm].mean(0), lab_i[..., 1:][ring].mean(0)
    dd = img_bgr.astype(float).mean(2) - base_bgr.astype(float).mean(2)
    spill = float(np.abs(dd[far] - np.median(dd[far])).mean()) if far.sum() > 100 else 0.0
    return {
        "vis": round(max(mean_change, float(dark_i - dark_b)), 2),
        "mean_change": round(mean_change, 2),
        "dark_gain": round(float(dark_i - dark_b), 2),
        "color": round(float(np.linalg.norm(ab_in - ab_ring)), 2),
        "shape_iou": round(inter / max(union, 1), 4),
        "mask_out": round(float((m & ~pill_b).sum() / m.sum()), 3),
        "spill": round(spill, 2),
        "area": int(m.sum()),
    }


def flags(cls, r, faint_cut):
    f = []
    if r["vis"] < faint_cut:
        f.append("faint")
    if r["color"] > MAX_COLOR:
        f.append("color")
    if r["shape_iou"] < MIN_SHAPE_IOU:
        f.append("deformed")
    if r["mask_out"] > MAX_MASK_OUT:
        f.append("off-pill")
    return f


def pct_rank(values, higher_is_worse=True):
    v = np.asarray(values, float)
    order = v.argsort().argsort() / max(len(v) - 1, 1)
    return order if higher_is_worse else 1 - order


def tile(img, mask, caption, size=240):
    m = (mask > 127).astype(np.uint8)
    full = img.copy()
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(full, cnts, -1, (0, 0, 255), 2)
    full = cv2.resize(full, (size, size))
    ys, xs = np.nonzero(m)
    cy, cx = int(ys.mean()), int(xs.mean())
    half = max(48, int(max(np.ptp(ys), np.ptp(xs)) * 0.75))
    y0, y1 = max(0, cy - half), min(img.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(img.shape[1], cx + half)
    zoom = cv2.resize(img[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_CUBIC)
    t = np.hstack([full, zoom])
    bar = np.full((26, t.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return np.vstack([bar, t])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--syn", required=True)
    ap.add_argument("--mvtec", required=True)
    ap.add_argument("--out", default=None, help="default: <syn>/qc")
    ap.add_argument("--faint", type=float, default=0.15, help="share of each class flagged as least visible")
    ap.add_argument("--top", type=float, default=0.0, help="extra share sent to review by overall score")
    ap.add_argument("--per-sheet", type=int, default=20)
    a = ap.parse_args()
    syn, out = Path(a.syn), Path(a.out or Path(a.syn) / "qc")
    out.mkdir(parents=True, exist_ok=True)

    base_of = {r["file"]: r["base_image"] for r in csv.DictReader(open(syn / "manifest.csv", encoding="utf-8"))}
    rows = []
    for cls in DEFECTS:
        files = sorted((syn / cls / "images").glob("*.png"))
        for f in files:
            mp = syn / cls / "masks" / f.name
            bp = Path(a.mvtec) / "train" / "good" / base_of.get(f.name, "")
            if not mp.exists() or not bp.is_file():
                continue
            img = cv2.imread(str(f))
            base = cv2.resize(cv2.imread(str(bp)), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            r = measures(img, base, cv2.imread(str(mp), 0))
            if r:
                rows.append({"class": cls, "file": f.name, **r})
        print(f"{cls}: scored {sum(r['class'] == cls for r in rows)}")

    suggested = []
    for cls in DEFECTS:
        cr = [r for r in rows if r["class"] == cls]
        if not cr:
            continue
        worst = np.max(np.vstack([
            pct_rank([r["vis"] for r in cr], higher_is_worse=False),
            pct_rank([r["color"] for r in cr]),
            pct_rank([r["shape_iou"] for r in cr], higher_is_worse=False),
            pct_rank([r["spill"] for r in cr]),
        ]), axis=0)
        cut = np.quantile(worst, 1 - a.top) if a.top > 0 else 2.0
        faint_cut = float(np.quantile([r["vis"] for r in cr], a.faint))
        for r, s in zip(cr, worst):
            r["score"] = round(float(s), 3)
            r["flags"] = ",".join(flags(cls, r, faint_cut))
            r["review"] = int(bool(r["flags"]) or s >= cut)
            if r["flags"]:
                suggested.append(f"{r['file']}  # {r['flags']}")
        review = sorted((r for r in cr if r["review"]), key=lambda r: (not r["flags"], r["vis"]))
        for k in range(0, len(review), a.per_sheet):
            tiles = []
            for r in review[k:k + a.per_sheet]:
                img = cv2.imread(str(syn / cls / "images" / r["file"]))
                mask = cv2.imread(str(syn / cls / "masks" / r["file"]), 0)
                cap = f"{r['file'].split('_')[-1][:-4]} vis{r['vis']:.0f} col{r['color']:.0f} {r['flags']}"
                tiles.append(tile(img, mask, cap))
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
            cv2.imwrite(str(out / f"suspects_{cls}_{k // a.per_sheet + 1:02d}.jpg"), grid,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{cls}: {sum(r['review'] for r in cr)} to review ({sum(bool(r['flags']) for r in cr)} flagged)")

    with open(out / "scores.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "blacklist_suggested.txt").write_text("\n".join(suggested) + "\n", encoding="utf-8")
    print(f"wrote {out / 'scores.csv'}, review sheets, and {len(suggested)} flagged files in blacklist_suggested.txt")


if __name__ == "__main__":
    main()

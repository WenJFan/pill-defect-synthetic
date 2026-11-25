"""Make a partially generated dataset safe to resume after a crash or disconnect.

generate_defects.py resumes by counting the images of each class and naming
the next one <class>_<count>.png. That only works if every class folder holds
an unbroken run 000..N-1 of readable images, each with its mask. A disconnect
can leave a half-written PNG, an image without its mask, or a manifest row
without a file.

For each class this keeps the longest clean run from 000, deletes everything
after the first problem, and rewrites manifest.csv to match the files on disk.

    python repair_partial.py --syn D:\\Work_Study\\pill_work\\pill_defects            # report only
    python repair_partial.py --syn D:\\Work_Study\\pill_work\\pill_defects --apply    # fix
"""
from __future__ import annotations

import argparse
import csv
import os
import re

from PIL import Image

DEFECTS = ["scratch", "crack", "contamination", "contamination_raised"]


def readable(path: str) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def clean_run(syn: str, cls: str) -> tuple[int, list[str]]:
    """Return (number of good files from 000, files to delete)."""
    img_dir, mask_dir = os.path.join(syn, cls, "images"), os.path.join(syn, cls, "masks")
    if not os.path.isdir(img_dir):
        return 0, []
    pat = re.compile(rf"^{re.escape(cls)}_(\d+)\.png$")
    idx = sorted(int(m.group(1)) for f in os.listdir(img_dir) if (m := pat.match(f)))
    good = 0
    for i in idx:
        name = f"{cls}_{i:03d}.png"
        if i != good or not readable(os.path.join(img_dir, name)):
            break
        if cls != "good" and not (os.path.exists(os.path.join(mask_dir, name))
                                  and readable(os.path.join(mask_dir, name))):
            break
        good += 1
    doomed = []
    for d in (img_dir, mask_dir):
        if os.path.isdir(d):
            for f in os.listdir(d):
                m = pat.match(f)
                if m and int(m.group(1)) >= good:
                    doomed.append(os.path.join(d, f))
    return good, doomed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--syn", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    keep: dict[str, int] = {}
    to_delete: list[str] = []
    for cls in DEFECTS:
        n, doomed = clean_run(args.syn, cls)
        keep[cls] = n
        to_delete += doomed
        print(f"{cls:22s} clean run: {n:4d}   to delete: {len(doomed)}")
    good_dir = os.path.join(args.syn, "good", "images")
    good_files = set(os.listdir(good_dir)) if os.path.isdir(good_dir) else set()
    bad_good = [os.path.join(good_dir, f) for f in good_files if not readable(os.path.join(good_dir, f))]
    to_delete += bad_good
    print(f"{'good':22s} files: {len(good_files):4d}   unreadable: {len(bad_good)}")

    mpath = os.path.join(args.syn, "manifest.csv")
    rows, header = [], ["class", "file", "base_image", "prompt"]
    if os.path.exists(mpath):
        with open(mpath, newline="", encoding="utf-8", errors="replace") as f:
            r = csv.reader(f)
            header = next(r, header)
            rows = [row for row in r if len(row) >= 2]

    def kept(row) -> bool:
        cls, fname = row[0], row[1]
        if cls == "good":
            return fname in good_files and os.path.join(good_dir, fname) not in bad_good
        m = re.match(rf"^{re.escape(cls)}_(\d+)\.png$", fname)
        return bool(m) and int(m.group(1)) < keep.get(cls, 0)

    seen, new_rows = set(), []
    for row in rows:
        if kept(row) and row[1] not in seen:
            seen.add(row[1])
            new_rows.append(row)
    missing = sum(keep.values()) - sum(1 for r in new_rows if r[0] != "good")
    print(f"manifest rows: {len(rows)} -> {len(new_rows)}"
          + (f"  ({missing} kept images have no manifest row; base image unknown for those)" if missing else ""))

    if not args.apply:
        print("\nReport only. Re-run with --apply to delete the files above and rewrite manifest.csv.")
        return
    for p in to_delete:
        os.remove(p)
    with open(mpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(new_rows)
    print(f"deleted {len(to_delete)} files; manifest rewritten. Safe to resume generate_defects.py.")


if __name__ == "__main__":
    main()

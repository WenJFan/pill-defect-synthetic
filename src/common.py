"""Shared helpers: pill segmentation and file listing."""
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def pill_mask(rgb, channel="gray"):
    """Binary mask (uint8 0/255) of the largest bright object (the pill).

    channel="gray" suits MVTec (light pill on dark background).
    channel="red" suits PillQC (pink pill on teal background).
    """
    if channel == "red":
        ch = rgb[..., 0]
    else:
        ch = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ch = cv2.GaussianBlur(ch, (5, 5), 0)
    _, th = cv2.threshold(ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        return th
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(lab == largest, 255, 0).astype(np.uint8)
    # fill holes (dark speckles inside the pill)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled


def pill_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)[:, 0, :]  # (N, 2) as x, y

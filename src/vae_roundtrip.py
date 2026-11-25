"""Pass images through the Stable Diffusion VAE (encode -> decode) and save them.

Why: every synthetic defect image went through the inpainting pipeline, so the whole
image (not only the defect) carries the VAE's reconstruction fingerprint (slight blur,
texture and colour shifts). Good images never did. A classifier can then separate
"generated" from "real" instead of "defect" from "good", and on real test photos it
calls almost everything good. Round-tripping the good training images gives both
classes the same fingerprint, so only the defect itself is left to learn.

Also used by diagnose_shortcut.py to test that explanation.

    python vae_roundtrip.py --src D:\\...\\data\\pill\\train\\good --dst D:\\...\\pill_defects\\good_vae\\images --prefix real_
    python vae_roundtrip.py --src D:\\...\\pill_defects\\good\\images --dst D:\\...\\pill_defects\\good_vae\\images --prefix syn_
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image

MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"   # same VAE as generate_defects.py
SIZE = 512                                                    # generation resolution


def load_vae(device):
    from diffusers import AutoencoderKL
    dtype = torch.float16 if device == "cuda" else torch.float32
    return AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=dtype).to(device).eval(), dtype


@torch.no_grad()
def roundtrip(vae, dtype, device, images: list[Image.Image]) -> list[Image.Image]:
    x = np.stack([np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC), np.float32) for im in images])
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device, dtype) / 127.5 - 1.0
    lat = vae.encode(x).latent_dist.mode()
    y = vae.decode(lat).sample.clamp(-1, 1)
    y = ((y + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in y]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae, dtype = load_vae(device)
    os.makedirs(a.dst, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(a.src, "*.png")))
    todo = [p for p in paths if not os.path.exists(os.path.join(a.dst, a.prefix + os.path.basename(p)))]
    for i in range(0, len(todo), a.batch):
        chunk = todo[i:i + a.batch]
        for p, im in zip(chunk, roundtrip(vae, dtype, device, [Image.open(p) for p in chunk])):
            im.save(os.path.join(a.dst, a.prefix + os.path.basename(p)))
        print(f"  {min(i + a.batch, len(todo))}/{len(todo)}", end="\r")
    print(f"\n{len(paths)} images -> {a.dst} ({len(paths) - len(todo)} already done)")


if __name__ == "__main__":
    main()

"""Generate class-specific synthetic pill defects (original AMS 563 generation script).

The prompts, negative prompts, per-class settings, mask generators and post-processing
below are the original code. Changes:
  * paths are command-line arguments
  * the model id runwayml/stable-diffusion-inpainting was removed from Hugging Face;
    default is stable-diffusion-v1-5/stable-diffusion-inpainting (same weights)
  * per-class lora_scale now takes effect. The original wrapped the UNet with PeftModel and
    passed cross_attention_kwargs={"scale": ...}, which PEFT adapters ignore, so every class
    ran at scale 1.0. Use --lora-scale-mode original to reproduce that behaviour.
  * --crack-masks: the original sampled crack shapes from MVTec ground_truth/crack, which are
    the masks of the TEST images used for evaluation. "mvtec_gt" reproduces that; "edge" draws
    synthetic edge-chip masks instead and keeps the test set unseen.
  * resumable (counts existing files), and writes manifest.csv (class, file, base image, prompt)

Output layout (as the original): <out>/<class>/images/*.png and <out>/<class>/masks/*.png

Usage:
  python src/generate_defects.py --mvtec data/pill --lora work/defectfill_lora_weights \
      --out work/pill_defects --crack-masks edge [--test 20]
"""
import argparse
import csv
import glob
import io
import os
import random

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

IMAGE_SIZE = (512, 512)

# ==========================================
# 2. Defect configs (original)
# ==========================================
GLOBAL_NEG_PROMPT = (
    "red, pink, purple, colorful, vibrant, bright color, mosaic, artifacts, "
    "distorted, torn, saturated, painted, cartoon, unrealistic texture, text, watermark"
)

ENV_PROMPTS = [
    "industrial inspection camera image",
    "macro inspection image",
    "low contrast lighting",
    "slightly noisy industrial camera image",
]

# ==========================================
# 2. Defect configs
# ==========================================
DEFECT_CONFIGS = {
    "scratch": {
        "mask_pool": None,
        "prompt": [
            "macro photo of a pill with a clearly visible curved surface scratch, narrow grey abrasion line, shallow coating damage, realistic tablet texture",
            "industrial inspection image of a tablet with a visible irregular scratch on the pill surface, thin abrasion groove, realistic coating defect",
            "macro photo of a pharmaceutical pill with a noticeable shallow scratch, curved narrow surface mark, slightly darker abrasion, intact pill shape"
        ],
        "neg_prompt": (
            f"deep hole, deep crack, missing chunk, edge chip, contamination, colorful stain, "
            f"painted streak, torn surface, shattered pill, black gouge, severe cut, blurry, {GLOBAL_NEG_PROMPT}"
        ),
        "lora_scale": 0.45,
        "brightness_factor": 0.94,
        "min_pixels": 180,
        "max_pixels": 700,
        "dilation": 3,
        "num": 650,
    },

    "crack": {
        "mask_pool": "ground_truth/crack",
        "prompt": [
            "macro photo of a pill with a small chipped edge, missing piece on the boundary, recessed grey broken area, realistic matte broken texture, intact overall pill shape",
            "industrial inspection image of a tablet with a minor edge chip, small broken corner, grey recessed damaged coating, realistic tablet surface",
            "macro photo of a pharmaceutical pill with a subtle chipped-off edge, shallow broken boundary defect, rough grey interior texture",
        ],
        "neg_prompt": (
            f"thin line crack, scratch, tree branch shape, branch-like line, contamination, colorful stain, "
            f"large missing area, shattered pill, oversaturated, blurry, painted, {GLOBAL_NEG_PROMPT}"
        ),
        "lora_scale": 0.52,
        "brightness_factor": 1.0,
        "min_pixels": 350,
        "max_pixels": 3500,
        "dilation": 1,
        "num": 650,
    },

    # 原来的 flat / smudge contamination，保留
    "contamination": {
        "mask_pool": None,
        "prompt": [
            "macro photo of a pill, slight surface contamination, subtle dirt, natural texture variation",
            "macro photo of a pill, faint grey smudge, micro-texture variation, dull residue",
        ],
        "neg_prompt": f"hole, crack, scratch, deep, recessed, 3d depth, sharp edges, {GLOBAL_NEG_PROMPT}",
        "lora_scale": 0.30,
        "brightness_factor": 0.85,
        "min_pixels": 1500,
        "num": 200,
    },

    # 新增：MVTec-like raised contamination / foreign material
    "contamination_raised": {
        "mask_pool": None,
        "prompt": [
            "macro photo of a pharmaceutical pill with a small raised chalky foreign material attached to the surface, off-white residue lump, visible 3D protrusion, soft contact shadow, realistic industrial defect",
            "industrial inspection image of a tablet with a tiny attached contaminant, pale residue bump on pill surface, raised foreign material, subtle highlight and contact shadow",
            "macro photo of a pill with a small off-white clump of foreign material stuck on the coating, protruding residue deposit, realistic tablet texture, visible volume",
            "macro photo of a pill with a raised residue particle attached near the tablet edge, light grey white foreign material, chalky texture, slight 3D bump",
            "industrial inspection image of a tablet with a small chalky deposit attached to the surface, slightly protruding foreign object, realistic defect with mild shadow",
            "macro photo of a pill with a moderately large attached foreign material deposit, off-white chalky lump, realistic raised contamination"
        ],
        "neg_prompt": (
            f"flat smudge, flat stain, dark stain, oil stain, crack, scratch, chip, deep hole, "
            f"colorful patch, red contamination, black contamination, melting, painted texture, "
            f"oversaturated, large melted blob, unrealistic giant lump, {GLOBAL_NEG_PROMPT}"
        ),
        "lora_scale": 0.40,
        "brightness_factor": 1.00,
        "min_pixels": 120,
        "max_pixels": 2200,   # 原来太小了，这里放宽
        "num": 300,
    },
}

# ==========================================
# 4. Mask utilities
# ==========================================
def get_pill_mask(base_image, threshold=35, shrink=0):
    pill_gray = base_image.convert("L")
    pill_mask = pill_gray.point(lambda p: 255 if p > threshold else 0)
    if shrink and shrink > 1:
        shrink = shrink if shrink % 2 == 1 else shrink + 1
        pill_mask = pill_mask.filter(ImageFilter.MinFilter(shrink))
    return pill_mask

def get_edge_band(pill_mask, band_size=55):
    band_size = band_size if band_size % 2 == 1 else band_size + 1
    inner_pill = pill_mask.filter(ImageFilter.MinFilter(band_size))
    return ImageChops.subtract(pill_mask, inner_pill)

def mask_area(mask, threshold=8):
    return int(np.sum(np.array(mask) > threshold))

def apply_dilation(mask, size):
    if not size or size <= 1:
        return mask
    size = size if size % 2 == 1 else size + 1
    return mask.filter(ImageFilter.MaxFilter(size))

def generate_centered_blob_mask(target_size=IMAGE_SIZE, min_pixels=1500):
    cx, cy = target_size[0] // 2, target_size[1] // 2
    while True:
        mask = Image.new("L", target_size, 0)
        draw = ImageDraw.Draw(mask)
        for _ in range(random.randint(1, 3)):
            rx, ry = random.randint(20, 70), random.randint(20, 70)
            ox, oy = random.randint(-30, 30), random.randint(-30, 30)
            draw.ellipse([cx + ox - rx, cy + oy - ry, cx + ox + rx, cy + oy + ry], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(5, 15)))
        mask = mask.point(lambda p: 255 if p > 128 else 0)
        if np.sum(np.array(mask)) // 255 > min_pixels:
            return mask

def generate_surface_scratch_mask(base_image, target_size=IMAGE_SIZE, min_pixels=180, max_pixels=700, dilation=3):
    pill_mask = get_pill_mask(base_image)
    safe_surface = pill_mask.filter(ImageFilter.MinFilter(45))

    arr = np.array(safe_surface)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None

    for _ in range(120):
        mask = Image.new("L", target_size, 0)
        draw = ImageDraw.Draw(mask)
        idx = random.randint(0, len(xs) - 1)
        x, y = int(xs[idx]), int(ys[idx])
        num_ctrl = random.randint(6, 9)
        step_len = random.randint(24, 42)
        angle = random.uniform(0, 2 * np.pi)
        points = [(x, y)]

        for _ in range(num_ctrl - 1):
            angle += random.uniform(-0.85, 0.85)
            x = x + int(np.cos(angle) * step_len) + random.randint(-5, 5)
            y = y + int(np.sin(angle) * step_len) + random.randint(-5, 5)
            points.append((x, y))

        base_width = random.randint(2, 3)
        for i in range(len(points) - 1):
            seg_width = max(1, base_width + random.choice([-1, 0, 0, 1]))
            draw.line([points[i], points[i + 1]], fill=255, width=seg_width)

        if random.random() < 0.35:
            offset_x = random.randint(-5, 5)
            offset_y = random.randint(-4, 4)
            aux_points = [(px + offset_x, py + offset_y) for px, py in points]
            for i in range(len(aux_points) - 1):
                draw.line([aux_points[i], aux_points[i + 1]], fill=130, width=1)

        mask = ImageChops.multiply(mask, safe_surface)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.4))

        if dilation > 1:
            d = dilation if dilation % 2 == 1 else dilation + 1
            mask = mask.filter(ImageFilter.MaxFilter(d))

        mask = mask.filter(ImageFilter.GaussianBlur(radius=0.55))
        area = np.sum(np.array(mask) > 8)
        if min_pixels <= area <= max_pixels and mask.getbbox():
            return mask
    return None

def generate_raised_contamination_mask(base_image, target_size=IMAGE_SIZE, min_pixels=120, max_pixels=2200):
    """
    MVTec-like contamination: compact raised residue / foreign material.
    Supports small / medium / large size modes.
    """
    pill_mask = get_pill_mask(base_image, shrink=5)
    edge_band = get_edge_band(pill_mask, band_size=75)
    surface_mask = pill_mask.filter(ImageFilter.MinFilter(41))

    # 约 50% 在边缘，50% 在表面
    use_edge = random.random() < 0.50
    anchor_mask = edge_band if use_edge else surface_mask

    arr = np.array(anchor_mask)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return None

    for _ in range(120):
        mask = Image.new("L", target_size, 0)
        draw = ImageDraw.Draw(mask)

        idx = random.randint(0, len(xs) - 1)
        cx, cy = int(xs[idx]), int(ys[idx])

        # ---------- 尺寸模式 ----------
        p = random.random()
        if p < 0.50:
            size_mode = "small"
        elif p < 0.85:
            size_mode = "medium"
        else:
            size_mode = "large"

        if size_mode == "small":
            rx = random.randint(10, 22)
            ry = random.randint(12, 28)
            extra_blob_prob = 0.35

        elif size_mode == "medium":
            rx = random.randint(18, 34)
            ry = random.randint(20, 40)
            extra_blob_prob = 0.55

        else:  # large
            rx = random.randint(28, 52)
            ry = random.randint(26, 58)
            extra_blob_prob = 0.80

        # 主块
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

        # 附属块，让形状不那么规则
        if random.random() < extra_blob_prob:
            n_extra = 1 if size_mode != "large" else random.randint(1, 2)
            for _ in range(n_extra):
                rx2 = random.randint(max(5, rx // 3), max(10, rx // 2))
                ry2 = random.randint(max(6, ry // 3), max(12, ry // 2))
                ox = random.randint(-rx, rx)
                oy = random.randint(-ry, ry)
                draw.ellipse([cx + ox - rx2, cy + oy - ry2, cx + ox + rx2, cy + oy + ry2], fill=210)

        # 大块时偶尔再加一点不规则边角
        if size_mode == "large" and random.random() < 0.45:
            poly = [
                (cx + random.randint(-rx, rx), cy + random.randint(-ry, ry)),
                (cx + random.randint(-rx, rx), cy + random.randint(-ry, ry)),
                (cx + random.randint(-rx, rx), cy + random.randint(-ry, ry)),
                (cx + random.randint(-rx, rx), cy + random.randint(-ry, ry)),
            ]
            draw.polygon(poly, fill=180)

        mask = ImageChops.multiply(mask, pill_mask)

        # 大块附着物边缘不要太糊
        blur_r = random.uniform(0.6, 1.0) if size_mode != "large" else random.uniform(0.4, 0.8)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_r))

        area = mask_area(mask)
        if min_pixels <= area <= max_pixels and mask.getbbox():
            return mask

    return None

def sample_real_mask_on_pill(base_image, mask_path, config, d_type, target_size=IMAGE_SIZE):
    if mask_path is None:
        return None

    pill_mask = get_pill_mask(base_image, shrink=0)
    edge_band = get_edge_band(pill_mask, band_size=65)

    raw_orig = Image.open(mask_path).convert("L").resize(target_size)
    raw_orig = raw_orig.point(lambda p: 255 if p > 20 else 0)
    raw_orig = apply_dilation(raw_orig, config.get("dilation", 1))

    for _ in range(120):
        raw = raw_orig.rotate(random.randint(0, 360), expand=False, fillcolor=0)
        new_mask = Image.new("L", target_size, 0)
        off_x = random.randint(-160, 160)
        off_y = random.randint(-160, 160)
        new_mask.paste(raw, (off_x, off_y))

        if d_type == "crack":
            candidate = ImageChops.multiply(new_mask, pill_mask)
            if not candidate.getbbox():
                continue

            area = mask_area(candidate)
            if area < config["min_pixels"] or area > config.get("max_pixels", 10**9):
                continue

            edge_overlap = ImageChops.multiply(candidate, edge_band)
            edge_area = mask_area(edge_overlap)
            if edge_area < 0.40 * area:
                continue

            inner_pill = pill_mask.filter(ImageFilter.MinFilter(85))
            inner_overlap = ImageChops.multiply(candidate, inner_pill)
            inner_area = mask_area(inner_overlap)
            if inner_area > 0.55 * area:
                continue

            final_mask = candidate.filter(ImageFilter.MaxFilter(3))
            final_mask = final_mask.filter(ImageFilter.GaussianBlur(radius=1.4))
            return final_mask

        elif d_type == "scratch":
            constrained = ImageChops.multiply(new_mask, get_pill_mask(base_image, shrink=7))
            final_mask = constrained.filter(ImageFilter.GaussianBlur(radius=0.45))
            area = mask_area(final_mask)
            if config["min_pixels"] <= area <= config.get("max_pixels", 10**9) and final_mask.getbbox():
                return final_mask
        else:
            final_mask = ImageChops.multiply(new_mask, pill_mask)
            area = mask_area(final_mask)
            if config["min_pixels"] <= area <= config.get("max_pixels", 10**9) and final_mask.getbbox():
                return final_mask

    return None

def get_industrial_mask_safe(base_image, mask_path, config, d_type, target_size=IMAGE_SIZE):
    pill_mask = get_pill_mask(base_image, threshold=35, shrink=0)

    if d_type == "contamination":
        for _ in range(100):
            raw_mask = generate_centered_blob_mask(target_size, config["min_pixels"])
            new_mask = Image.new("L", target_size, 0)
            new_mask.paste(raw_mask, (random.randint(-150, 150), random.randint(-150, 150)))
            final_mask = ImageChops.multiply(new_mask, pill_mask)
            if final_mask.getbbox() and (np.sum(np.array(final_mask)) // 255 > config["min_pixels"]):
                return final_mask.filter(ImageFilter.GaussianBlur(radius=5.0))
        return None

    if d_type == "contamination_raised":
        return generate_raised_contamination_mask(
            base_image,
            target_size=target_size,
            min_pixels=config["min_pixels"],
            max_pixels=config["max_pixels"],
        )

    if d_type in ["scratch", "crack"]:
        real_mask = sample_real_mask_on_pill(base_image, mask_path, config, d_type, target_size)
        if real_mask is not None:
            return real_mask
        if d_type == "scratch":
            return generate_surface_scratch_mask(
                base_image,
                target_size=target_size,
                min_pixels=config["min_pixels"],
                max_pixels=config["max_pixels"],
                dilation=config["dilation"],
            )
        return None

    return None

# ==========================================
# 5. Post-processing
# ==========================================
def desaturate_mask_area(img, mask, blur_radius=3):
    img_arr = np.array(img).astype(np.float32)
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask_arr = np.array(soft_mask) / 255.0
    gray = np.mean(img_arr, axis=2, keepdims=True)
    img_arr = img_arr * (1 - mask_arr[..., None]) + gray * mask_arr[..., None]
    return Image.fromarray(np.clip(img_arr, 0, 255).astype(np.uint8))

def apply_local_brightness(image, mask, factor):
    if factor == 1.0:
        return image
    blur_r = 12.0 if factor < 0.9 else 4.0
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return Image.composite(ImageEnhance.Brightness(image).enhance(factor), image, soft_mask)

def enhance_scratch_visibility(image, mask, darken=0.86, contrast=1.20):
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))
    darker = ImageEnhance.Brightness(image).enhance(darken)
    darker = ImageEnhance.Contrast(darker).enhance(contrast)
    return Image.composite(darker, image, soft_mask)

def enhance_crack_visibility(image, mask, darken=0.72, contrast=1.12):
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=1.6))
    darker = ImageEnhance.Brightness(image).enhance(darken)
    darker = ImageEnhance.Contrast(darker).enhance(contrast)
    return Image.composite(darker, image, soft_mask)

def shift_mask(mask, dx, dy):
    shifted = Image.new("L", mask.size, 0)
    shifted.paste(mask, (dx, dy))
    return shifted

def enhance_raised_contamination(image, mask):
    """
    Make attached contaminant look more raised:
    - contact shadow at lower-right
    - highlight at upper-left
    - brighter compact core
    """
    # 核心、阴影、高光三个层次
    core_mask = mask.filter(ImageFilter.GaussianBlur(radius=1.0))
    shadow_mask = shift_mask(mask.filter(ImageFilter.GaussianBlur(radius=3.0)), 3, 3)
    highlight_mask = shift_mask(mask.filter(ImageFilter.GaussianBlur(radius=2.0)), -2, -2)

    out = image

    # 1) 接触阴影：让异物像“贴”在药片上
    shadow_img = ImageEnhance.Brightness(out).enhance(0.76)
    shadow_img = ImageEnhance.Contrast(shadow_img).enhance(1.08)
    out = Image.composite(shadow_img, out, shadow_mask)

    # 2) 高光：模拟顶部受光
    highlight_img = ImageEnhance.Brightness(out).enhance(1.16)
    highlight_img = ImageEnhance.Contrast(highlight_img).enhance(1.06)
    out = Image.composite(highlight_img, out, highlight_mask)

    # 3) 核心提亮：让异物主体更“厚”
    core_img = ImageEnhance.Brightness(out).enhance(1.10)
    core_img = ImageEnhance.Contrast(core_img).enhance(1.10)
    out = Image.composite(core_img, out, core_mask)

    return out

def add_lighting_variation(img):
    if random.random() < 0.35:
        img_arr = np.array(img).astype(np.float32)
        color_shift = np.random.uniform(0.93, 1.07, size=(3,))
        img_arr = np.clip(img_arr * color_shift, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
    if random.random() < 0.35:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.82, 1.18))
    return img

def add_realism(img):
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    if random.random() < 0.3:
        img_arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(2, 6), img_arr.shape)
        img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
    if random.random() < 0.3:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=random.randint(50, 85))
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")
    return img

def perturb_mask(mask, d_type=None):
    if d_type == "crack":
        if random.random() < 0.35:
            return mask.filter(ImageFilter.MaxFilter(random.choice([3, 5])))
        return mask
    if d_type in ["scratch", "contamination_raised"]:
        return mask
    if random.random() < 0.5:
        r = random.choice([3, 5, 7])
        if random.random() < 0.5:
            mask = mask.filter(ImageFilter.MaxFilter(r))
        else:
            mask = mask.filter(ImageFilter.MinFilter(r))
    return mask


def generate_edge_chip_mask(base_image, config, target_size=IMAGE_SIZE):
    """Test-set-free alternative for crack masks: a small, jagged bite centred exactly on the
    pill outline (real MVTec cracks are small dark breaks on the edge), clipped to the pill."""
    pill_mask = get_pill_mask(base_image, shrink=0)
    outline = get_edge_band(pill_mask, band_size=7)  # a thin ring on the pill boundary
    ys, xs = np.where(np.array(outline) > 0)
    if len(xs) == 0:
        return None
    for _ in range(120):
        mask = Image.new("L", target_size, 0)
        draw = ImageDraw.Draw(mask)
        i = random.randint(0, len(xs) - 1)
        cx, cy = int(xs[i]), int(ys[i])
        rx, ry = random.randint(14, 32), random.randint(9, 20)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
        for _ in range(random.randint(1, 3)):  # jagged outline
            ox, oy = random.randint(-rx, rx), random.randint(-ry, ry)
            r2 = random.randint(4, 10)
            draw.ellipse([cx + ox - r2, cy + oy - r2, cx + ox + r2, cy + oy + r2], fill=255)
        mask = mask.rotate(random.randint(0, 360), center=(cx, cy))
        candidate = ImageChops.multiply(mask, pill_mask)
        area = mask_area(candidate)
        if area < config["min_pixels"] or area > config.get("max_pixels", 10**9):
            continue
        if mask_area(ImageChops.multiply(candidate, outline)) == 0:
            continue
        return candidate.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(radius=1.4))
    return None


# ==========================================
# 6. Generation loop
# ==========================================
def set_lora_scale(unet, scale):
    from peft.tuners.lora import LoraLayer
    for m in unet.modules():
        if isinstance(m, LoraLayer):
            for name in m.scaling:
                m.set_scale(name, scale)


def load_pipe(model_id, lora_path, device):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
    from peft import PeftModel

    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                          safety_checker=None).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    if lora_path:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, os.path.abspath(lora_path))
    pipe.set_progress_bar_config(disable=True)
    return pipe


def jpeg(img, lo, hi):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=random.randint(lo, hi))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec", required=True, help="MVTec 'pill' folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--crack-masks", choices=["edge", "mvtec_gt"], default="edge")
    ap.add_argument("--lora-scale-mode", choices=["per_class", "original"], default="per_class",
                    help="original = scale 1.0 for every class, as the first run effectively did")
    ap.add_argument("--classes", nargs="*", default=None, help="subset of: " + ", ".join(DEFECT_CONFIGS) + ", good")
    ap.add_argument("--test", type=int, default=None, help="quick trial: this many images per class")
    ap.add_argument("--num-good", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    good_images = sorted(glob.glob(os.path.join(args.mvtec, "train", "good", "*.png")))
    if not good_images:
        raise RuntimeError(f"No good images under {args.mvtec}/train/good")
    os.makedirs(args.out, exist_ok=True)
    classes = args.classes or list(DEFECT_CONFIGS) + ["good"]

    mpath = os.path.join(args.out, "manifest.csv")
    new = not os.path.exists(mpath)
    mf = open(mpath, "a", newline="")
    mw = csv.writer(mf)
    if new:
        mw.writerow(["class", "file", "base_image", "prompt"])

    pipe = None
    for d_type in [c for c in classes if c in DEFECT_CONFIGS]:
        if pipe is None:
            print("Loading DefectFill pipeline...")
            pipe = load_pipe(args.model, args.lora, args.device)
        config = DEFECT_CONFIGS[d_type]
        num = args.test or config["num"]
        img_out = os.path.join(args.out, d_type, "images")
        mask_out = os.path.join(args.out, d_type, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)
        if args.lora:
            set_lora_scale(pipe.unet, 1.0 if args.lora_scale_mode == "original" else config["lora_scale"])

        mask_pool = None
        if config["mask_pool"] and args.crack_masks == "mvtec_gt":
            mask_pool = glob.glob(os.path.join(args.mvtec, config["mask_pool"], "*.png"))

        made = len(glob.glob(os.path.join(img_out, "*.png")))  # resume
        print(f"Batch processing: {d_type.upper()} ({num} samples, {made} already done)...")
        attempts, max_attempts = 0, max(num * 20, 1)
        while made < num and attempts < max_attempts:
            attempts += 1
            base_path = random.choice(good_images)
            init_img = Image.open(base_path).convert("RGB").resize(IMAGE_SIZE)
            current_prompt = f"{random.choice(config['prompt'])}, {random.choice(ENV_PROMPTS)}"

            if d_type == "crack" and args.crack_masks == "edge":
                mask_img = generate_edge_chip_mask(init_img, config)
            else:
                mask_path = random.choice(mask_pool) if mask_pool else None
                mask_img = get_industrial_mask_safe(init_img, mask_path, config, d_type)
            if mask_img is None:
                continue

            if d_type == "scratch":
                steps, cfg, strength = 28, 5.8, 0.78
            elif d_type == "crack":
                steps, cfg, strength = 30, 6.0, 0.86
            elif d_type == "contamination_raised":
                steps, cfg, strength = 30, 6.2, 0.88
            else:
                steps, cfg, strength = 25, 5.0, 0.85

            output = pipe(prompt=current_prompt, negative_prompt=config["neg_prompt"], image=init_img,
                          mask_image=mask_img, height=IMAGE_SIZE[1], width=IMAGE_SIZE[0],
                          num_inference_steps=steps, guidance_scale=cfg, strength=strength).images[0]

            if d_type == "contamination":
                output = desaturate_mask_area(output, mask_img, blur_radius=3)
            elif d_type == "crack":
                output = desaturate_mask_area(output, mask_img, blur_radius=2)
            elif d_type == "contamination_raised":
                output = desaturate_mask_area(output, mask_img, blur_radius=1.0)

            final_output = apply_local_brightness(output, mask_img, config["brightness_factor"])
            if d_type == "scratch":
                final_output = enhance_scratch_visibility(final_output, mask_img, darken=0.86, contrast=1.20)
            elif d_type == "crack":
                final_output = enhance_crack_visibility(final_output, mask_img, darken=0.72, contrast=1.15)
            elif d_type == "contamination_raised":
                final_output = enhance_raised_contamination(final_output, mask_img)

            final_output = add_lighting_variation(final_output)
            if d_type == "contamination":
                final_output = add_realism(final_output)
            elif d_type == "contamination_raised" and random.random() < 0.15:
                final_output = jpeg(final_output, 78, 92)
            elif d_type == "crack" and random.random() < 0.10:
                final_output = jpeg(final_output, 78, 92)

            fname = f"{d_type}_{made:03d}.png"
            final_output.save(os.path.join(img_out, fname))
            perturb_mask(mask_img, d_type=d_type).save(os.path.join(mask_out, fname))
            mw.writerow([d_type, fname, os.path.basename(base_path), current_prompt])
            mf.flush()
            made += 1
            if made % 10 == 0:
                print(f"  [{d_type}] Progress: {made}/{num}")
        if made < num:
            print(f"Only generated {made}/{num} for {d_type}. Check mask settings.")

    if "good" in classes:
        # Normal samples alignment (original): real good images with lighting / blur / noise / JPEG
        print("Aligning normal data distribution...")
        good_dir = os.path.join(args.out, "good", "images")
        os.makedirs(good_dir, exist_ok=True)
        n = args.test or args.num_good
        for i in range(n):
            fname = f"good_{i:03d}.png"
            if os.path.exists(os.path.join(good_dir, fname)):
                continue
            path = random.choice(good_images)
            img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
            add_realism(add_lighting_variation(img)).save(os.path.join(good_dir, fname))
            mw.writerow(["good", fname, os.path.basename(path), ""])
    mf.close()
    print("Done! Dataset generation completed.")


if __name__ == "__main__":
    main()

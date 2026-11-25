# Pill Defect Detection via Diffusion-Based Synthetic Data

Real defective pills are rare, so a defect classifier has little to learn from. This project
generates localized pill defects with Stable Diffusion inpainting (plus a DefectFill-style LoRA)
on real good pills, then tests whether a classifier trained **only on synthetic defects** detects
**real** defects.

**Key finding.** The first classifiers scored a perfect F1 = 1.0 on synthetic validation data and
close to chance on real photos (AUROC 0.56). A controlled test showed why: they had learned the Stable
Diffusion VAE's fingerprint ("is this image generated?"), not the defects. After giving the good
training images the same fingerprint, the same ResNet18 reaches **AUROC 0.947 ± 0.016** on the real
MVTec test set (3 seeds). That beats an autoencoder trained on real good pills (0.749), and no real
defect image is used for training.

Started as a course project for AMS 563 (Medical Image Analysis) at Stony Brook University, spring 2026,
then re-run end to end and extended with the shortcut diagnosis and fix.

## Pipeline

```
MVTec good pills ─► masks ─► SD inpainting + LoRA ─► post-processing ─► QC ─► classifiers ─► real MVTec test
(train/good, 267)            LoRA: DefectFill on      lighting, noise,   auto + ranked   B1 / B2 / B3
                             PillQC chip images       blur, JPEG         manual review
```

| Step | Script |
|---|---|
| Chip masks for PillQC (repo ships images only) | `src/make_chip_masks.py` |
| DefectFill LoRA: token `sks`, UNet + text-encoder LoRA r=8, L_def + L_obj, 2000 steps | `src/train_lora_defectfill.py` |
| Class-specific generation (crack, scratch, contamination, raised contamination, good) | `src/generate_defects.py` |
| Make an interrupted generation run safe to resume | `src/repair_partial.py` |
| QC: drop near-unchanged images; rank the rest by visibility, colour, shape and mask placement | `src/clean.py`, `src/qc_rank.py` |
| Baseline 1: conv autoencoder on real good only | `src/baseline_1_autoencoder.py` |
| Baseline 2: synthetic-only 4-class ResNet18 | `src/baseline_2_synthetic_multiclass.py` |
| Baseline 3: binary ResNet18 trained on synthetic data, tested on real | `src/baseline_3_binary.py` |
| Shortcut test and fix: SD-VAE round trip | `src/diagnose_shortcut.py`, `src/vae_roundtrip.py` |

## Data (not included)

- **MVTec AD `pill`** (CC BY-NC-SA 4.0). Generation base: train/good (267). Real test sets:
  **targeted** = 26 good + 71 defects (crack 26, scratch 24, contamination 21), the three types we
  synthesise; **all** = 26 good + 141 defects (every type, including color, imprint, pill type).
- **PillQC** ([matlab-deep-learning/pillQC](https://github.com/matlab-deep-learning/pillQC)): 43 chip
  images; 34 get an automatic mask and train the LoRA.

Generated set after QC: crack 639, scratch 588, contamination 182, raised contamination 300,
good 1000. QC removed 46 images automatically (no visible change) and 45 after manual review of the
260 least visible ones flagged by `qc_rank.py`.

## Results (re-run, November 2026)

Real MVTec test, **targeted** set. Flagging every image as defective already gives F1 0.845, so
F1 must beat that. AUROC is the main comparison. F1 is at the fixed threshold 0.5 (or B1's
training-data threshold), never tuned on the test set.

| Model | Training data | Seeds | AUROC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| B1 autoencoder | 267 real good | 1 | 0.749 | 0.364 | 0.941 | 0.225 |
| B3 v1 | real good + synthetic defects | 1 | 0.559 | 0.000 | – | 0.000 |
| B3 v2 | + 1000 synthetic good | 1 | 0.684 | 0.000 | – | 0.000 |
| B3 v3_aug | v2 + strong augmentation | 3 | 0.558 ± 0.040 | 0.834 ± 0.014 | 0.733 | 0.967 |
| B3 v3_aug_nosyngood | v3_aug without synthetic good | 3 | 0.526 ± 0.030 | 0.763 ± 0.076 | 0.724 | 0.817 |
| **B3 v4_vae** | **v2, good images passed through the SD VAE** | 3 | **0.947 ± 0.016** | **0.909 ± 0.027** | 0.904 | 0.915 |
| B3 v4_vae_aug | v4_vae + strong augmentation | 3 | 0.887 ± 0.038 | 0.880 ± 0.031 | 0.811 | 0.967 |

On the **all** set (141 defects, including 4 types never synthesised), v4_vae reaches AUROC
0.918 ± 0.015 and F1 0.900 ± 0.020; B1 reaches 0.751. Per-seed numbers and both test sets are in
[`results/summary.json`](results/summary.json).

B2 (4-class, synthetic train and synthetic test): macro F1 0.988 with a random split and 0.994 with
a split by base image. The generated classes are distinct, but this says nothing about real photos.

### The shortcut, and the test that found it

Every synthetic defect image passes through Stable Diffusion's VAE, which slightly changes the
*whole* image (texture, sharpness, colour). Good images did not. `diagnose_shortcut.py` takes the
real test photos, passes them through the VAE (the pill is unchanged), and scores them again:

| Model | Real good images flagged: original → after VAE round trip |
|---|---|
| B3 v1 | 0% → **100%** |
| B3 v2 | 0% → **100%** |
| B3 v3_aug_nosyngood | 46% → **100%** |
| B3 v4_vae (fix) | 27% → 46% |

The earlier models label any image with the VAE fingerprint as defective, whatever the pill looks
like. This explains near-perfect synthetic validation scores alongside near-chance real AUROC.
Adding synthetic good images (v2) or heavier augmentation (v3) did not remove it. The fix
(`vae_roundtrip.py`) passes all good training images through the same VAE, so the fingerprint no
longer separates the classes. Some sensitivity remains (27% → 46%), so the fix reduces the
shortcut but does not remove it.

### How to read the numbers

- **Small test set.** 26 good images, so one image is about 4 points of specificity. v4_vae flags
  27% ± 12% of good pills at threshold 0.5 (specificity 0.73).
- **Crack masks.** Generation used `--crack-masks mvtec_gt`: crack *shapes* come from the MVTec test
  ground truth (the test images themselves are never used). Excluding cracks, v4_vae's AUROC is
  0.968 ± 0.019 (good vs scratch + contamination), so the result does not depend on them. Cracks alone
  give 0.912 ± 0.018. The `--crack-masks edge` default avoids test data entirely.
- **Model selection** uses a synthetic validation split, and thresholds are fixed at 0.5: nothing is
  tuned on the test set.
- **Seeds matter.** Runs of the same version with different seeds differed by up to 0.19 AUROC,
  so v3 and v4 are reported over 3 seeds; v1 and v2 (one run each) are kept for the shortcut test.

## Run

Local GPU (tested on an RTX 5070 Ti, Windows; needs the CUDA 12.8 PyTorch build) or
`notebooks/run_all.ipynb` on Colab.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cd src
$W="D:\pill_work"; $D="$W\data"; $S="$W\pill_defects"; $R="$W\runs"
python make_chip_masks.py --pillqc $D\pillQC --out $W\pillqc_chip
python train_lora_defectfill.py --data $W\pillqc_chip --out $W\defectfill_lora_weights
python generate_defects.py --mvtec $D\pill --lora $W\defectfill_lora_weights --out $S --crack-masks mvtec_gt
python clean.py auto --syn $S --mvtec $D\pill
python qc_rank.py --syn $S --mvtec $D\pill          # review $S\qc\suspects_*.jpg, write a blacklist
python clean.py apply --syn $S --blacklist $W\blacklist.txt
python vae_roundtrip.py --src $D\pill\train\good --dst $S\good_vae\images --prefix real_
python vae_roundtrip.py --src $S\good\images --dst $S\good_vae\images --prefix syn_
foreach ($seed in 1,2,3) { python baseline_3_binary.py --version v4_vae --mvtec $D\pill --syn $S --out "$R\b3_v4_vae_s$seed" --seed $seed }
python diagnose_shortcut.py --mvtec $D\pill --runs $R\b3_v4_vae_s1
```

## Fixes compared with the original code

1. `runwayml/stable-diffusion-inpainting` was removed from Hugging Face; the default is
   `stable-diffusion-v1-5/stable-diffusion-inpainting` (same weights).
2. LoRA weights were trained in pure float16; now float32 weights with fp16 autocast.
3. Per-class `lora_scale` never took effect (PEFT ignores `cross_attention_kwargs`); it is now
   applied. `--lora-scale-mode original` reproduces the old behaviour.
4. `--crack-masks edge` (default) avoids test-set crack masks; `mvtec_gt` reproduces the original.
5. Baseline 2 can split by base image (`--group-split`).
6. Baseline 3 picks the best epoch on synthetic validation only; every run reports AUROC,
   the fixed-threshold F1 and the flag-everything floor on both real test sets.

## Limitations

- 26 good test images: specificity estimates are coarse, and 73% means about 1 in 4 good pills is flagged.
- Generated cracks are mostly flat grey patches at the pill edge; generated contamination is subtle.
- The VAE round trip reduces but does not remove the generation shortcut.
- One product (MVTec pill); no evidence yet for other pill types or imaging setups.

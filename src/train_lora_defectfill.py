"""DefectFill-style LoRA fine-tuning of SD inpainting on PillQC chip images.

Faithful to the original AMS 563 script:
  * placeholder token "sks", LoRA r=8 on the UNet (to_q/k/v/out) and the text encoder
  * L_def on the defect mask ("A photo of sks") + L_obj with an adjusted mask
    ("A pill with sks"), total = 0.5 * L_def + 0.2 * L_obj
  * 2000 steps, batch 2, lr 2e-4 (UNet) / 4e-5 (text encoder)

Fixes against the original:
  * runwayml/stable-diffusion-inpainting was removed from Hugging Face; default is now
    stable-diffusion-v1-5/stable-diffusion-inpainting (same weights)
  * the original trained LoRA weights in pure float16 (NaN risk); here trainable weights stay
    float32 and the forward pass runs under fp16 autocast with a grad scaler
  * the text-encoder LoRA is saved too (the original saved only the UNet LoRA)

Input: the folder written by make_chip_masks.py (metadata.jsonl + images/ + masks/).

Usage:
  python src/train_lora_defectfill.py --data work/pillqc_chip --out work/defectfill_lora_weights
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DefectDataset(Dataset):
    def __init__(self, root, size=512):
        self.root = Path(root)
        self.rows = [json.loads(l) for l in open(self.root / "metadata.jsonl")]
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.latent = size // 8

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        image = self.tf(Image.open(self.root / r["image"]).convert("RGB"))
        mask = transforms.ToTensor()(Image.open(self.root / r["mask"]).convert("L"))
        mask = transforms.Resize((self.latent, self.latent))(mask)  # latent space (512 / 8)
        return {"pixel_values": image, "masks": mask}


def train(pipe, dataset, args, device):
    from peft import LoraConfig, get_peft_model

    use_amp = device == "cuda"
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    pipe.tokenizer.add_tokens(args.token)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    if use_amp:
        pipe.unet.enable_gradient_checkpointing()

    unet = get_peft_model(pipe.unet, LoraConfig(r=args.rank, target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    text_encoder = get_peft_model(pipe.text_encoder,
                                  LoraConfig(r=args.rank, target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    opt = torch.optim.AdamW([
        {"params": [p for p in unet.parameters() if p.requires_grad], "lr": args.lr_unet},
        {"params": [p for p in text_encoder.parameters() if p.requires_grad], "lr": args.lr_text},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=False)
    sf = pipe.vae.config.scaling_factor
    vae_dtype = next(pipe.vae.parameters()).dtype

    def encode(prompts):
        ids = pipe.tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True,
                             max_length=pipe.tokenizer.model_max_length).input_ids.to(device)
        return text_encoder(ids)[0]

    print(f"Starting DefectFill training: {len(dataset)} pairs, {args.steps} steps")
    unet.train()
    text_encoder.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            bs = batch["pixel_values"].shape[0]
            with torch.no_grad():
                latents = pipe.vae.encode(batch["pixel_values"].to(device, vae_dtype)).latent_dist.sample() * sf
            latents = latents.float()
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()
            noisy = noise_scheduler.add_noise(latents, noise, t)
            mask = batch["masks"].to(device)
            masked_latents = latents * (1 - mask)  # as in the original
            unet_in = torch.cat([noisy, mask, masked_latents], dim=1)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                # L_def: defect concept, loss only inside the mask
                c_def = encode([f"A photo of {args.token}"] * bs)
                pred_def = unet(unet_in, t, c_def).sample
                # L_obj: defect on the object, background down-weighted (0.3)
                c_obj = encode([f"A {args.object} with {args.token}"] * bs)
                pred_obj = unet(unet_in, t, c_obj).sample
            loss_def = F.mse_loss(mask * pred_def.float(), mask * noise)
            adj = mask + 0.3 * (1 - mask)
            loss_obj = F.mse_loss(adj * pred_obj.float(), adj * noise)
            loss = 0.5 * loss_def + 0.2 * loss_obj

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"Step {step}/{args.steps} - Loss: {loss.item():.4f}")
            if step >= args.steps:
                break
    return unet, text_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--token", default="sks")
    ap.add_argument("--object", default="pill")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr-unet", type=float, default=2e-4)
    ap.add_argument("--lr-text", type=float, default=4e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from diffusers import StableDiffusionInpaintPipeline

    torch.manual_seed(args.seed)
    device = "cuda"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model, torch_dtype=torch.float32, safety_checker=None)
    pipe.to(device)
    pipe.vae.to(dtype=torch.float16)  # frozen, encode only
    unet, text_encoder = train(pipe, DefectDataset(args.data), args, device)

    os.makedirs(args.out, exist_ok=True)
    unet.save_pretrained(args.out)  # same layout as the original (loaded with PeftModel.from_pretrained)
    text_encoder.save_pretrained(os.path.join(args.out, "text_encoder_lora"))
    json.dump(vars(args), open(os.path.join(args.out, "train_args.json"), "w"), indent=2)
    print(f"Training finished. Weights saved to {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Handwritten OCR (Local / Offline) with TrOCR
============================================
- Runs fully **locally** on your machine (CPU or GPU).
- Works **offline** when you load a model from a local folder (see README).
- TrOCR is state-of-the-art for **handwritten** text compared to classic OCR.

Install (once):
    pip install -r requirements.txt

Quick start (online first run to fetch weights into a local folder):
    # 1) Download model weights to ./models/trocr-base-handwritten
    huggingface-cli download microsoft/trocr-base-handwritten \
        --local-dir ./models/trocr-base-handwritten --local-dir-use-symlinks False

Run (completely offline after that):
    python handwriting_ocr.py --image /path/to/image_or_folder \
        --model-dir ./models/trocr-base-handwritten --offline

Notes:
- If you do NOT pass --model-dir, the script will try to pull from Hugging Face by model id
  (requires internet the first time). To keep everything offline, always use --model-dir.
- For best accuracy on messy handwriting, try the larger model and good scans.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
        return [path]
    if path.is_dir():
        out = []
        for ext in SUPPORTED_EXTS:
            out.extend(path.rglob(f"*{ext}"))
        return sorted(out)
    return []


def load_rgb_image(p: Path) -> Image.Image:
    img = Image.open(p)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_trocr(model_dir: str = None, model_id: str = "microsoft/trocr-base-handwritten", device: str = "cpu"):
    """
    Load TrOCR from a local folder (preferred for offline), or by model id (online on first run).
    """
    if model_dir:
        processor = TrOCRProcessor.from_pretrained(model_dir)
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    else:
        processor = TrOCRProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)

    model.to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def ocr_image(
    image: Image.Image,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: str,
    max_new_tokens: int = 96,
    num_beams: int = 4,
) -> str:
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated = model.generate(
        pixel_values,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description="Local/offline handwritten OCR with TrOCR.")
    ap.add_argument("--image", required=True, help="Path to an image file or a directory of images.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--model-dir", type=str, default=None, help="Path to a LOCAL TrOCR model directory (offline).")
    group.add_argument("--model-id", type=str, default="microsoft/trocr-base-handwritten",
                       help="Hugging Face model id (requires internet on first run).")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Force device; default auto-detect.")
    ap.add_argument("--max-new-tokens", type=int, default=96, help="Max generated text length.")
    ap.add_argument("--num-beams", type=int, default=4, help="Beam search width.")
    ap.add_argument("--offline", action="store_true", help="Force HuggingFace offline mode (no network).")
    ap.add_argument("--fp16", action="store_true", help="Use half precision on CUDA to save VRAM.")
    args = ap.parse_args()

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.fp16 and device != "cuda":
        print("[!] --fp16 ignored (CUDA not available).", file=sys.stderr)

    # Offline mode (prevents accidental downloads)
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Also disable HF Hub token/network
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Sanity for offline + no local model
    if args.offline and not args.model_dir:
        print("[!] Offline mode requested but no --model-dir provided. "
              "Point --model-dir to a local TrOCR folder (see README).", file=sys.stderr)
        sys.exit(2)

    # Load model
    try:
        processor, model = load_trocr(args.model_dir, args.model_id, device)
        if args.fp16 and device == "cuda":
            model.half()
    except Exception as e:
        print(f"[!] Failed to load model: {e}", file=sys.stderr)
        sys.exit(3)

    # Gather images
    images = find_images(Path(args.image))
    if not images:
        print(f"[!] No images found at: {args.image}", file=sys.stderr)
        sys.exit(1)

    # OCR loop
    for p in images:
        try:
            img = load_rgb_image(p)
            text = ocr_image(
                img, processor, model, device,
                max_new_tokens=args.max_new_tokens, num_beams=args.num_beams
            )
            print(f"\n=== {p} ===\n{text}\n")
        except Exception as e:
            print(f"[!] Error processing {p}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

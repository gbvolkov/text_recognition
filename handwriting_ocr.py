#!/usr/bin/env python3
"""
Handwritten OCR (Local/Offline) — TrOCR, robust loader

- Works fully offline with a local model folder (pass --model-dir and --offline).
- Avoids "You need to specify either text or text_target" by splitting image processor & tokenizer.
- Good for English handwriting; swap model dir for a Cyrillic/Russian-finetuned TrOCR to read Russian.

Install (once):
    pip install pillow torch transformers huggingface_hub

Download model locally (one-time):
    hf download akushsky/trocr-large-handwritten-ru --local-dir ./models/trocr-large-handwritten

Example (fully offline):
    python handwriting_ocr.py --image ./data/i.jpg \
      --model-dir ./models/trocr-large-handwritten --offline --use-fast \
      --resize-longest 1280 --contrast 1.2 --sharpen 1.05 \
      --num-beams 8 --max-new-tokens 256
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image, ImageOps, ImageEnhance

# Transformers imports (split components)
from transformers import (
    AutoTokenizer,
    AutoImageProcessor,    # preferred modern API
    VisionEncoderDecoderModel,
)

# Fallbacks for older transformers versions (only used if AutoImageProcessor missing)
try:
    from transformers import AutoFeatureExtractor as _AutoFeatureExtractor
except Exception:
    _AutoFeatureExtractor = None

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


def preprocess_image(
    img: Image.Image,
    resize_longest: Optional[int] = None,
    contrast: float = 1.0,
    sharpen: float = 1.0,
) -> Image.Image:
    # Optional: resize to help small handwriting
    if resize_longest and resize_longest > 0:
        w, h = img.size
        longest = max(w, h)
        if longest != resize_longest:
            if w >= h:
                new_w = resize_longest
                new_h = int(h * (resize_longest / w))
            else:
                new_h = resize_longest
                new_w = int(w * (resize_longest / h))
            img = img.resize((new_w, new_h), Image.LANCZOS)

    # Optional: mild contrast/sharpening
    if abs(contrast - 1.0) > 1e-3:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if abs(sharpen - 1.0) > 1e-3:
        img = ImageEnhance.Sharpness(img).enhance(sharpen)
    return img


def _load_image_processor(source: str, local_only: bool):
    """
    Robustly load an image processor without triggering text-only processors.
    Prefers AutoImageProcessor; falls back to AutoFeatureExtractor on older versions.
    """
    try:
        return AutoImageProcessor.from_pretrained(source, local_files_only=local_only)
    except Exception as e:
        if _AutoFeatureExtractor is None:
            raise
        # Older transformers: use (deprecated) feature extractor
        return _AutoFeatureExtractor.from_pretrained(source, local_files_only=local_only)


def load_components(
    model_dir: Optional[str],
    model_id: str,
    device: str,
    use_fast: bool,
    offline: bool,
):
    local_only = offline or bool(model_dir)
    source = model_dir if model_dir else model_id

    # Enforce offline if requested
    if offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Load tokenizer + image processor + model separately
    image_processor = _load_image_processor(source, local_only=local_only)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=use_fast, local_files_only=local_only)
    model = VisionEncoderDecoderModel.from_pretrained(source, local_files_only=local_only)

    model.to(device)
    model.eval()

    # Helpful banner
    print(f"[i] Image processor: {source} (local_only={local_only})", file=sys.stderr)
    print(f"[i] Tokenizer:       {source} (use_fast={use_fast}, local_only={local_only})", file=sys.stderr)
    print(f"[i] Model:           {source} (device={device})", file=sys.stderr)
    return image_processor, tokenizer, model


@torch.inference_mode()
def ocr_image(
    image: Image.Image,
    image_processor,
    tokenizer,
    model: VisionEncoderDecoderModel,
    device: str,
    max_new_tokens: int = 128,
    num_beams: int = 6,
) -> str:
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
    generated = model.generate(
        pixel_values,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return text.strip()


def main():
    ap = argparse.ArgumentParser(description="Local/offline handwritten OCR (TrOCR)")
    ap.add_argument("--image", required=True, help="Path to an image file or a directory of images.")

    group = ap.add_mutually_exclusive_group()
    group.add_argument("--model-dir", type=str, default=None, help="LOCAL model directory (preferred for offline).")
    group.add_argument("--model-id", type=str, default="microsoft/trocr-large-handwritten",
                       help="Model id (first run needs internet if not cached).")

    ap.add_argument("--offline", action="store_true", help="Force offline mode (no network).")
    ap.add_argument("--use-fast", action="store_true", help="Use fast tokenizer (recommended).")

    ap.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Force device; default auto-detect.")
    ap.add_argument("--fp16", action="store_true", help="Use half precision on CUDA to save VRAM.")

    ap.add_argument("--max-new-tokens", type=int, default=128, help="Max generated text length.")
    ap.add_argument("--num-beams", type=int, default=6, help="Beam search size.")

    ap.add_argument("--resize-longest", type=int, default=None, help="Resize so longest side equals this (e.g., 1280).")
    ap.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier (e.g., 1.2).")
    ap.add_argument("--sharpen", type=float, default=1.0, help="Sharpness multiplier (e.g., 1.05).")

    args = ap.parse_args()

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.fp16 and device != "cuda":
        print("[!] --fp16 ignored (CUDA not available).", file=sys.stderr)

    if args.offline and not args.model_dir:
        print("[!] --offline requires --model-dir pointing to a local model folder.", file=sys.stderr)
        sys.exit(2)

    try:
        image_processor, tokenizer, model = load_components(
            model_dir=args.model_dir,
            model_id=args.model_id,
            device=device,
            use_fast=bool(args.use_fast),
            offline=bool(args.offline),
        )
        if args.fp16 and device == "cuda":
            model.half()
    except Exception as e:
        print(f"[!] Failed to load components: {e}", file=sys.stderr)
        sys.exit(3)

    images = find_images(Path(args.image))
    if not images:
        print(f"[!] No images found at: {args.image}", file=sys.stderr)
        sys.exit(1)

    for p in images:
        try:
            img = load_rgb_image(p)
            img = preprocess_image(img, args.resize_longest, args.contrast, args.sharpen)
            text = ocr_image(
                img, image_processor, tokenizer, model, device,
                max_new_tokens=args.max_new_tokens, num_beams=args.num_beams
            )
            print(f"\n=== {p} ===\n{text}\n")
        except Exception as e:
            print(f"[!] Error processing {p}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
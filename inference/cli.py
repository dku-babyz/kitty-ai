"""Small CLI helper to run text or image inference from the terminal.

Examples
--------
$ python -m inference.cli --text "욕설이 포함된 예시"
$ python -m inference.cli --image ./cat.png
"""
import argparse, sys, json
from pathlib import Path
from . import TextPredictor, ImagePredictor

def main():
    parser = argparse.ArgumentParser(description="Unified text/image predictor")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--image", type=str, help="Path to image file to classify")
    parser.add_argument("--text_model_dir", type=Path, default=Path("inference/text/model"), help="Directory with fine‑tuned text model")
    parser.add_argument("--image_ckpt", type=Path, default=Path("inference/image/model.ckpt"), help="Path to Lightning checkpoint for image model")
    args = parser.parse_args()

    if not args.text and not args.image:
        parser.error("At least one of --text or --image must be provided.")

    if args.text:
        predictor = TextPredictor(args.text_model_dir)
        print(json.dumps(predictor(args.text), ensure_ascii=False, indent=2))
    if args.image:
        predictor = ImagePredictor(args.image_ckpt)
        (label,), ms = predictor(args.image, return_proba=False)
        print(f"{args.image} ➜ {label}   [{ms:.1f} ms]")

if __name__ == "__main__":
    main()
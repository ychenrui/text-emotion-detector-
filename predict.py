"""Stage 4 — Inference: top-3 emotions with calibrated probabilities.

Loads the Stage-2 classifier and the Stage-3 temperature, then for any input
message returns the top-k labels and their calibrated sigmoid probabilities.

CLI:
    python predict.py "I can't believe she actually said yes!"
    python predict.py            # interactive REPL
    python predict.py -k 5 "..." # change top-k

Library:
    from predict import EmotionClassifier
    clf = EmotionClassifier()
    clf.predict("I'm so tired of this.")
    # -> [('annoyance', 0.71), ('sadness', 0.42), ('disappointment', 0.31)]
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    CALIBRATION_FILE,
    CLASSIFIER_OUTPUT_DIR,
    GOEMOTIONS_LABELS,
    MAX_SEQ_LENGTH,
)
from utils import get_device


class EmotionClassifier:
    def __init__(self, model_path: str | None = None, calibration_file: str | None = None):
        model_path = model_path or str(CLASSIFIER_OUTPUT_DIR / "final")
        calibration_file = calibration_file or str(CALIBRATION_FILE)

        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

        try:
            with open(calibration_file) as f:
                self.temperature = float(json.load(f)["temperature"])
        except FileNotFoundError:
            print(f"[predict] WARNING: no calibration at {calibration_file}; using T=1.0")
            self.temperature = 1.0

    @torch.no_grad()
    def predict(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(self.device)
        logits = self.model(**enc).logits[0]
        probs = torch.sigmoid(logits / self.temperature).cpu().numpy()
        order = probs.argsort()[::-1][:top_k]
        return [(GOEMOTIONS_LABELS[int(i)], float(probs[int(i)])) for i in order]

    @torch.no_grad()
    def predict_batch(self, texts: list[str], top_k: int = 3) -> list[list[tuple[str, float]]]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
        ).to(self.device)
        logits = self.model(**enc).logits
        probs = torch.sigmoid(logits / self.temperature).cpu().numpy()
        out = []
        for row in probs:
            order = row.argsort()[::-1][:top_k]
            out.append([(GOEMOTIONS_LABELS[int(i)], float(row[int(i)])) for i in order])
        return out


def _print_results(text: str, results: list[tuple[str, float]]) -> None:
    print(f"\nInput: {text!r}")
    for label, prob in results:
        bar = "#" * int(prob * 40)
        print(f"  {label:>15s}  {prob:6.4f}  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-3 emotion classifier")
    parser.add_argument("text", nargs="?", help="Message to classify")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="How many emotions to return")
    args = parser.parse_args()

    clf = EmotionClassifier()

    if args.text:
        _print_results(args.text, clf.predict(args.text, top_k=args.top_k))
        return

    print("Interactive mode — enter a message (Ctrl+C / Ctrl+D to quit):")
    try:
        while True:
            text = input("> ").strip()
            if not text:
                continue
            _print_results(text, clf.predict(text, top_k=args.top_k))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()

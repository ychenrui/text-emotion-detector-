"""Stage 3 — Temperature scaling for calibrated multi-label probabilities.

Multi-label sigmoid heads are typically over- or under-confident: the raw
sigmoid value isn't a true probability of the label being correct. Temperature
scaling fits a single scalar T on a held-out set so that
    p = sigmoid(logits / T)
better matches empirical frequencies. We pick T by minimising BCE on the
GoEmotions validation set with LBFGS.

Run:
    python calibrate.py
"""
from __future__ import annotations

import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from config import CALIBRATION_FILE, CLASSIFIER_OUTPUT_DIR
from data_goemotions import load_goemotions
from utils import get_device


@torch.no_grad()
def collect_logits_and_labels(model, dataset, tokenizer, device, batch_size: int = 16):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = dataset.with_format("torch")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)

    all_logits, all_labels = [], []
    model.eval()
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        all_logits.append(out.logits.detach().cpu().float())
        all_labels.append(labels.detach().cpu().float())
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """LBFGS over a single scalar T. Returns the fitted T."""
    T = nn.Parameter(torch.ones(1))
    bce = nn.BCEWithLogitsLoss()
    opt = optim.LBFGS([T], lr=0.05, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = bce(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def main() -> None:
    device = get_device()
    print(f"[calibrate] device: {device}")

    model_path = CLASSIFIER_OUTPUT_DIR / "final"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained classifier at {model_path}. Run train_classifier.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)

    bundle = load_goemotions(tokenizer=tokenizer)
    val = bundle["data"]["validation"]

    logits, labels = collect_logits_and_labels(model, val, tokenizer, device)
    print(f"[calibrate] collected {logits.shape[0]} validation examples")

    pre_bce = nn.BCEWithLogitsLoss()(logits, labels).item()
    T = fit_temperature(logits, labels)
    post_bce = nn.BCEWithLogitsLoss()(logits / T, labels).item()

    print(f"[calibrate] T = {T:.4f}")
    print(f"[calibrate] BCE before: {pre_bce:.4f}")
    print(f"[calibrate] BCE after : {post_bce:.4f}")

    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(
            {"temperature": T, "pre_bce": pre_bce, "post_bce": post_bce},
            f,
            indent=2,
        )
    print(f"[calibrate] saved -> {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()

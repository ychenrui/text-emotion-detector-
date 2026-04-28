"""Stage 2 — Multi-label classification on GoEmotions.

Loads the MLM-adapted backbone from Stage 1 if it exists, attaches a 28-way
sigmoid head, and trains with BCEWithLogitsLoss + per-class pos_weight to
counter GoEmotions' label imbalance.

Run:
    python train_classifier.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import (
    CLASSIFIER_OUTPUT_DIR,
    CLS_CONFIG,
    GOEMOTIONS_LABELS,
    MLM_OUTPUT_DIR,
    MODEL_NAME,
    NUM_LABELS,
    SEED,
)
from data_goemotions import compute_pos_weight, load_goemotions
from utils import get_device, set_seed


class WeightedBCETrainer(Trainer):
    """Trainer that uses per-label pos_weight in BCEWithLogitsLoss."""

    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        if self._pos_weight is not None:
            pw = self._pos_weight.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics():
    """Threshold @ 0.5 macro/micro F1 — sanity metrics, not the production rule.
    Production uses top-3 with calibrated probs (see predict.py)."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.int32)
        labels_int = labels.astype(np.int32)
        return {
            "f1_macro": f1_score(labels_int, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(labels_int, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(labels_int, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels_int, preds, average="macro", zero_division=0),
        }

    return compute_metrics


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"[train_classifier] device: {device}")

    # Use the MLM-adapted backbone from Stage 1 if present.
    adapted = MLM_OUTPUT_DIR / "final"
    if adapted.exists() and (adapted / "config.json").exists():
        backbone = str(adapted)
        print(f"[train_classifier] loading MLM-adapted backbone: {backbone}")
    else:
        backbone = MODEL_NAME
        print(
            f"[train_classifier] no MLM checkpoint at {adapted}; "
            f"falling back to base model ({MODEL_NAME})."
        )

    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)

    id2label = {i: name for i, name in enumerate(GOEMOTIONS_LABELS)}
    label2id = {name: i for i, name in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )

    bundle = load_goemotions(tokenizer=tokenizer)
    data = bundle["data"]

    pos_weight = torch.tensor(compute_pos_weight(data["train"]))
    print(f"[train_classifier] pos_weight range: "
          f"{pos_weight.min().item():.2f} .. {pos_weight.max().item():.2f}")

    CLASSIFIER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(CLASSIFIER_OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=CLS_CONFIG["num_train_epochs"],
        learning_rate=CLS_CONFIG["learning_rate"],
        per_device_train_batch_size=CLS_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CLS_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CLS_CONFIG["gradient_accumulation_steps"],
        warmup_ratio=CLS_CONFIG["warmup_ratio"],
        weight_decay=CLS_CONFIG["weight_decay"],
        logging_steps=CLS_CONFIG["logging_steps"],
        save_steps=CLS_CONFIG["save_steps"],
        eval_steps=CLS_CONFIG["eval_steps"],
        save_total_limit=CLS_CONFIG["save_total_limit"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=CLS_CONFIG["load_best_model_at_end"],
        metric_for_best_model=CLS_CONFIG["metric_for_best_model"],
        greater_is_better=True,
        # Same precision story as Stage 1: bf16 mixed precision on the 4090.
        fp16=False,
        bf16=True,
        bf16_full_eval=True,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=SEED,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedBCETrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
        pos_weight=pos_weight,
    )

    trainer.train()

    final_dir = CLASSIFIER_OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("[train_classifier] === final test-set evaluation ===")
    test_metrics = trainer.evaluate(eval_dataset=data["test"])
    print(json.dumps(test_metrics, indent=2))
    with open(final_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"[train_classifier] saved classifier -> {final_dir}")


if __name__ == "__main__":
    main()

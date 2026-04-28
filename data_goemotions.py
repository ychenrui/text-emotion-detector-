"""Load GoEmotions ('simplified' config) and prepare for multi-label classification.

Each example becomes:
    input_ids, attention_mask : tokenised text
    labels                    : multi-hot float vector of length NUM_LABELS

The training loop then applies BCEWithLogitsLoss across the 28 labels.
"""
from __future__ import annotations

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from config import MAX_SEQ_LENGTH, MODEL_NAME, NUM_LABELS


def load_goemotions(tokenizer=None):
    """Return {'data': DatasetDict (train/validation/test), 'tokenizer': ...}.

    The 'simplified' config already discards rare/multi-rater disagreement
    cases and stores labels as a list of integer ids.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    raw = load_dataset("go_emotions", "simplified")

    def to_multi_hot(example):
        vec = [0.0] * NUM_LABELS
        for idx in example["labels"]:
            if 0 <= idx < NUM_LABELS:
                vec[idx] = 1.0
        return {"labels": vec}

    raw = raw.map(to_multi_hot)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    # Drop original columns we don't need; keep 'labels'.
    drop_cols = [c for c in ("text", "id") if c in raw["train"].column_names]
    tokenised = raw.map(tokenize, batched=True, remove_columns=drop_cols)

    return {"data": tokenised, "tokenizer": tokenizer}


def compute_pos_weight(train_ds) -> np.ndarray:
    """Per-label pos_weight for BCEWithLogitsLoss.

    GoEmotions is heavily imbalanced (~30% of examples are 'neutral', some
    emotions appear in <1% of rows). pos_weight = neg_count / pos_count
    rebalances the gradient signal so rare labels still get learned.
    """
    labels = np.array(train_ds["labels"], dtype=np.float32)
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    pos_weight = neg / np.clip(pos, 1.0, None)
    # Cap to avoid extreme values destabilising training.
    return np.clip(pos_weight, 1.0, 50.0).astype(np.float32)

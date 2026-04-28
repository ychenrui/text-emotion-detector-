"""Load Empathetic Dialogues and prepare it for MLM continued pretraining.

We don't use the emotion label here — the goal is *domain adaptation*: teach
DeBERTa what conversational, emotionally-expressive English looks like.
We assemble a 'situation prompt + utterance' string per row, dedupe, and
tokenise. The MLM loss is applied later by the data collator.
"""
from __future__ import annotations

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from config import MAX_SEQ_LENGTH, MODEL_NAME


def _row_to_text(row: dict) -> str:
    """Combine the situation prompt and the utterance into one string."""
    prompt = (row.get("prompt") or "").strip()
    utterance = (row.get("utterance") or "").strip()
    # ED stores comma-separated underscores like "i_was_so_excited"; clean those.
    utterance = utterance.replace("_comma_", ",")
    prompt = prompt.replace("_comma_", ",")
    if prompt and utterance and prompt != utterance:
        return f"{prompt} {utterance}"
    return utterance or prompt


def _build_text_examples(split) -> list[str]:
    seen = set()
    out = []
    for row in split:
        text = _row_to_text(row)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def load_ed_for_mlm(tokenizer=None):
    """Return tokenised train + validation datasets ready for MLM training."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # The HF script loader needs trust_remote_code on newer datasets versions.
    raw = load_dataset("empathetic_dialogues", trust_remote_code=True)

    train_texts = _build_text_examples(raw["train"])
    val_texts = _build_text_examples(raw["validation"])
    print(f"[data_ed] train texts after dedupe: {len(train_texts)}")
    print(f"[data_ed] val   texts after dedupe: {len(val_texts)}")

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,  # the collator will pad
        )

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    return {"train": train_tok, "validation": val_tok, "tokenizer": tokenizer}

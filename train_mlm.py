"""Stage 1 — Domain-adaptive MLM pretraining on Empathetic Dialogues.

Why MLM and not classification? We want the *backbone* to learn what
conversational, emotional English looks like before we attach the GoEmotions
head. Training on ED's single-label emotion targets here would corrupt the
representations for the multi-label task we actually care about.

Run:
    python train_mlm.py
"""
from __future__ import annotations

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import MLM_CONFIG, MLM_OUTPUT_DIR, MODEL_NAME, SEED
from data_ed import load_ed_for_mlm
from utils import get_device, set_seed


def main() -> None:
    set_seed(SEED)
    device = get_device()
    print(f"[train_mlm] device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    data = load_ed_for_mlm(tokenizer=tokenizer)

    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_CONFIG["mlm_probability"],
    )

    MLM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(MLM_OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=MLM_CONFIG["num_train_epochs"],
        learning_rate=MLM_CONFIG["learning_rate"],
        per_device_train_batch_size=MLM_CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=MLM_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=MLM_CONFIG["gradient_accumulation_steps"],
        warmup_ratio=MLM_CONFIG["warmup_ratio"],
        weight_decay=MLM_CONFIG["weight_decay"],
        logging_steps=MLM_CONFIG["logging_steps"],
        save_steps=MLM_CONFIG["save_steps"],
        eval_steps=MLM_CONFIG["eval_steps"],
        save_total_limit=MLM_CONFIG["save_total_limit"],
        eval_strategy="steps",
        save_strategy="steps",
        # RTX 4090 (Ada) supports bf16 natively — gives ~4x throughput vs fp32
        # and halves activation memory. Stick with bf16 over fp16 to avoid
        # loss-scaling fiddliness with mixed-precision Adam.
        fp16=False,
        bf16=True,
        bf16_full_eval=True,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    final_dir = MLM_OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[train_mlm] saved adapted backbone -> {final_dir}")


if __name__ == "__main__":
    main()

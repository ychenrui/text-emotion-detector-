# Emotion Classifier

Top-3 emotion classification for conversational text, built by fine-tuning
[`microsoft/deberta-v3-large`](https://huggingface.co/microsoft/deberta-v3-large)
on [GoEmotions](https://huggingface.co/datasets/go_emotions) with
[Empathetic Dialogues](https://huggingface.co/datasets/empathetic_dialogues)
as a domain-adaptation pretraining step. Outputs are temperature-scaled so the
returned probabilities are calibrated, not just rank-ordered.

```text
> I can't believe she actually said yes!
        joy  0.8421  ################################
  amusement  0.6133  ########################
   surprise  0.4107  ################
```

## Approach

A four-stage pipeline. Each stage is its own script; intermediate outputs are
written to `checkpoints/` so you can re-run any stage in isolation.

| Stage | Script | What it does |
|---|---|---|
| 1. Domain adaptation | `train_mlm.py` | Continued **MLM pretraining** on Empathetic Dialogues. Adapts the backbone to conversational/emotional language without imposing ED's single-label structure on the model. |
| 2. Multi-label classification | `train_classifier.py` | Loads the adapted backbone, attaches a 28-way sigmoid head, fine-tunes on GoEmotions with **BCEWithLogitsLoss + per-class `pos_weight`** to counter heavy label imbalance. |
| 3. Calibration | `calibrate.py` | Fits a single scalar **temperature `T`** on the validation set (LBFGS) so `sigmoid(logits / T)` matches empirical frequencies. Saves `T` to `checkpoints/temperature.json`. |
| 4. Inference | `predict.py` | Returns the top-3 emotions and their calibrated probabilities for any input message. |

### Why MLM (not classification) for Stage 1?

Empathetic Dialogues uses **single-label** emotion targets over a slightly
different label set than GoEmotions. Training a classification head on ED
first and then swapping the head for GoEmotions would either (a) waste the
head we just trained, or (b) push the backbone toward single-emotion
predictions, hurting top-k quality. Masked Language Modeling (the standard
[DAPT recipe](https://arxiv.org/abs/2004.10964)) lets the backbone soak up
conversational patterns without contaminating it with the wrong label
structure.

### Why temperature scaling?

Multi-label sigmoid heads trained with `pos_weight`-rebalanced BCE are
typically over-confident — the raw sigmoid value isn't a good probability.
A single scalar `T` (Platt scaling's simpler cousin) is enough to fix this
and costs almost nothing to fit.

## Project layout

```
emotion_classifier/
├── README.md
├── requirements.txt
├── config.py                 # Hyperparameters, paths, GoEmotions label list
├── utils.py                  # Device detection (CUDA → MPS → CPU), seeding
├── data_ed.py                # Empathetic Dialogues loader for MLM
├── data_goemotions.py        # GoEmotions loader + pos_weight computation
├── train_mlm.py              # Stage 1
├── train_classifier.py       # Stage 2
├── calibrate.py              # Stage 3
├── predict.py                # Stage 4 (CLI + EmotionClassifier class)
└── checkpoints/              # created during training
    ├── stage1_mlm/final/
    ├── stage2_classifier/final/
    └── temperature.json
```

## Setup

The code targets a single **RTX 4090** (24 GB VRAM, bf16). It also runs on
Apple Silicon (MPS) and CPU as a fallback, but the hyperparameters in
`config.py` are sized for the 4090 — drop the batch sizes if you're on
weaker hardware.

```bash
python -m venv .venv
source .venv/bin/activate

# Install the CUDA build of torch first (adjust cu121 → your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Datasets are downloaded automatically from the Hugging Face Hub on first run.

## Usage

Run the four stages in order:

```bash
python train_mlm.py            # Stage 1 — ~45 min on 4090
python train_classifier.py     # Stage 2 — ~30 min on 4090
python calibrate.py            # Stage 3 — minutes
python predict.py "..."        # Stage 4
```

Inference, both ways:

```bash
# One-shot CLI
python predict.py "I'm so tired of this."

# Interactive REPL
python predict.py

# Change top-k
python predict.py -k 5 "Wait, what just happened?"
```

As a library:

```python
from predict import EmotionClassifier

clf = EmotionClassifier()
clf.predict("I miss them so much.")
# -> [('sadness', 0.78), ('grief', 0.41), ('disappointment', 0.22)]

clf.predict_batch(["good morning!", "ugh fine, whatever."])
```

## Configuration

All hyperparameters live in `config.py`:

| Knob | Stage 1 (MLM) | Stage 2 (classifier) |
|---|---|---|
| Epochs | 3 | 5 (best checkpoint by `f1_macro`) |
| LR | 5e-5 | 2e-5 |
| Per-device batch | 16 | 32 |
| Grad accumulation | 1 | 1 |
| Warmup ratio | 0.06 | 0.06 |
| Weight decay | 0.01 | 0.01 |
| Max seq length | 128 | 128 |
| Precision | bf16 | bf16 |

If you OOM on a smaller GPU, halve `per_device_train_batch_size` and double
`gradient_accumulation_steps` to keep the effective batch the same.

## Notes & caveats

- **`f1_macro` at threshold 0.5** is reported during Stage 2 evaluation as a
  sanity metric. Your *actual* production criterion is "is the right emotion
  in the top 3", which `predict.py` answers directly. Don't get too anxious
  about absolute F1 numbers — multi-label F1@0.5 on GoEmotions is famously
  modest (~0.45–0.50 macro is competitive).
- **`pos_weight` is capped at 50×** in `data_goemotions.py` to prevent rare
  labels (`grief`, `relief`, `pride`) from destabilising training.
- **`neutral` is included as a 28th label** and frequently lands in the top-3.
  If you don't want it, filter it out at inference time rather than removing
  it during training — it's a useful "none of the above" signal.
- **Dataset loading** uses `trust_remote_code=True` for Empathetic Dialogues,
  which still uses a script-based loader on the Hub.

## Optional speed wins

These aren't enabled by default to keep dependencies minimal, but if you want
the classifier to train in ~20 min instead of ~30 min:

```python
# pip install flash-attn --no-build-isolation
model = AutoModelForSequenceClassification.from_pretrained(
    backbone, ..., attn_implementation="flash_attention_2",
)

# And/or after model creation:
model = torch.compile(model)
```

## License & attribution

- Backbone: [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) (MIT)
- Datasets: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Apache 2.0), [Empathetic Dialogues](https://github.com/facebookresearch/EmpatheticDialogues) (CC-BY-NC 4.0 — non-commercial)

Note that EmpatheticDialogues is **non-commercial use only**. If you're
shipping this in a product, retrain Stage 1 on a commercially-licensed
conversational corpus or skip it (the model still works without Stage 1, just
with a small quality hit).

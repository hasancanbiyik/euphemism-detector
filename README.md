# Euphemism Detector

A multilingual euphemism detection system built on fine-tuned XLM-RoBERTa. Given a sentence and a target phrase, the model predicts whether the phrase is used **euphemistically or literally** in context, with a confidence score.

---

## What is this?

Euphemisms are context-dependent — *"passed away"* signals death in one sentence and nothing unusual in another. This project trains a cross-lingual transformer model on labeled **Potentially Euphemistic Terms (PETs)** to learn those contextual signals, and serves predictions through a clean web interface.

Built as part of ongoing NLP research on cross-lingual euphemism detection.

---

## Project Structure

```
euphemism-detector/
├── v1-streamlit/       # Prototype — English only, Streamlit UI
├── v2-fastapi/         # Production UI — FastAPI + dark minimal frontend
├── v3-multilingual/    # Multilingual training — EN / TR / ES / ZH / YO
└── extension/          # Chrome extension (in development)
```

---

## Progress

### v1 — English Prototype (`v1-streamlit/`)

- Fine-tuned `xlm-roberta-base` on ~3,100 English PET examples
- Binary classification: euphemistic (1) vs literal (0)
- Target phrase marked with `[PET_BOUNDARY]` tokens — e.g. `"My grandfather [PET_BOUNDARY]passed away[PET_BOUNDARY] last Tuesday."`
- Added `[PET_BOUNDARY]` as a special token so the model attends to the target span directly
- Stratified 80/10/10 train/val/test split
- AdamW optimizer with weight decay, gradient clipping, linear warmup schedule
- Early stopping (patience=2) to prevent overfitting
- MPS support for Apple Silicon (M-series chips)
- Streamlit UI: sentence input + phrase input → prediction + confidence bars

**Results (English test set):**
| | Precision | Recall | F1 |
|---|---|---|---|
| Literal | 0.81 | 0.83 | 0.82 |
| Euphemistic | 0.88 | 0.86 | 0.87 |
| **Macro avg** | **0.84** | **0.84** | **0.84** |

---

### v2 — FastAPI + Custom UI (`v2-fastapi/`)

- Replaced Streamlit with a **FastAPI** backend exposing `/predict` and `/health` REST endpoints
- Built a custom **dark minimal frontend** in pure HTML/CSS/JS (no framework, single file)
  - Typography: Syne + DM Mono
  - Animated confidence bars
  - Session history with click-to-reload
  - Language selector (EN / TR / ES / ZH / YO)
  - Example sentences to try
  - Model input transparency expander
- Swapping the underlying model requires changing one line in `app.py`

---

### v3 — Multilingual Training (`v3-multilingual/`)

- Extended to **5 languages** using warm-start fine-tuning from the English model
- Combined dataset: ~14,300 rows across EN / TR / ES / ZH / YO
- `WeightedRandomSampler` to handle 64/36 euphemistic/literal class imbalance
- Training script **auto-detects hardware** and sets batch size + LR accordingly:

| Device | Batch size | Learning rate |
|--------|-----------|---------------|
| CUDA (cluster) | 32 | 1e-5 |
| MPS (Apple Silicon) | 8 | 2e-5 |
| CPU | 8 | 2e-5 |

- Per-language F1 breakdown printed at end of training
- Epochs increased to 10, patience to 3 vs v1
- Training currently running on HPC cluster

---

## Model

- **Base:** `xlm-roberta-base` (multilingual, 100 languages)
- **Task:** Sequence classification — euphemistic / literal
- **Input:** Full paragraph with `[PET_BOUNDARY]phrase[PET_BOUNDARY]` markers
- **Output:** Label + softmax confidence scores for both classes

Model weights are not included due to file size. Train using the scripts in each version folder.

---

## Data Format

| Column | Description |
|--------|-------------|
| `text` | Paragraph with `[PET_BOUNDARY]` markers around the target phrase |
| `label` | `1` = euphemistic, `0` = literal |
| `PET` | Extracted target phrase |
| `sentence` | Cleaned sentence without markers |
| `category` | Semantic category (e.g. death, employment, body) |
| `euph_status` | `always_euph` / `sometimes_euph` |

---

## Quickstart

```bash
git clone https://github.com/yourusername/euphemism-detector.git
cd euphemism-detector/v2-fastapi

pip install -r requirements.txt

# Train
python train.py --data your_dataset.csv --output ./model

# Run
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000`

---

## Roadmap

- [x] English fine-tuning — 84% macro F1
- [x] Custom web UI (FastAPI + HTML/CSS/JS)
- [x] Multilingual training — EN / TR / ES / ZH / YO
- [ ] Multilingual evaluation results
- [ ] Chrome extension for in-browser detection
- [ ] HuggingFace Hub model release
- [ ] Multimodal input support

---

## Research Context

Part of broader research on cross-lingual euphemism detection and transfer learning between Turkish and English.

*When Semantic Overlap Is Not Enough: Cross-Lingual Euphemism Transfer Between Turkish and English* (under review)

---

## License

MIT

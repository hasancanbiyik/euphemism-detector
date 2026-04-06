---
title: Euphemism Detector
emoji: 🔍
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
---

# Euphemism Detector

[![CI](https://github.com/hasancanbiyik/euphemism-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/hasancanbiyik/euphemism-detector/actions/workflows/ci.yml)

A multilingual euphemism detection API powered by fine-tuned XLM-RoBERTa. Given a sentence and a target phrase, the model predicts whether the phrase is used **euphemistically or literally** in context, returning confidence scores for both classes.

**Live demo:** [HuggingFace Spaces](https://huggingface.co/spaces/hasancanbiyik/euphemism-detector)

---

## Why This Exists

Euphemisms are context-dependent — *"passed away"* signals death in one sentence and means nothing unusual in another. Standard sentiment or toxicity classifiers miss this because euphemism is a pragmatic phenomenon, not a lexical one. This project trains a cross-lingual transformer to detect euphemistic usage across five languages by learning contextual signals around **Potentially Euphemistic Terms (PETs)**.

Built as part of ongoing NLP research on cross-lingual euphemism detection and transfer learning between Turkish and English.

---

## Architecture

```text
┌─────────────────────────────────────────────────┐
│                   Client                        │
│               (Browser / curl)                  │
└────────────────────┬────────────────────────────┘
                     │ HTTP
                     ▼
┌─────────────────────────────────────────────────┐
│             FastAPI Application                 │
│                                                 │
│  POST /predict                                  │
│    ├── Input validation                         │
│    ├── [PET_BOUNDARY] marking around phrase     │
│    ├── XLMRobertaTokenizer encoding             │
│    └── Softmax → label + confidence scores      │
│                                                 │
│  GET /health                                    │
│    └── Model status check                       │
│                                                 │
│  Static file serving (HTML/CSS/JS frontend)     │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│     XLM-RoBERTa (Sequence Classification)       │
│                                                 │
│  Base: xlm-roberta-base (100 languages)         │
│  Task: Binary — euphemistic (1) vs literal (0)  │
│  Input: Text with [PET_BOUNDARY] span markers   │
│  Vocab: 250,003 (base + PET_BOUNDARY token)     │
│  Weights: HuggingFace Hub (auto-downloaded)     │
└─────────────────────────────────────────────────┘

---

## Quickstart

### Run with Docker

```bash
docker build -t euphemism-detector .
docker run -p 7860:7860 euphemism-detector
```

Model weights are automatically downloaded from HuggingFace Hub on first startup. To use local weights instead:

```bash
docker run -p 7860:7860 -v ./model:/app/model euphemism-detector
```

### Run locally

```bash
git clone [https://github.com/hasancanbiyik/euphemism-detector.git](https://github.com/hasancanbiyik/euphemism-detector.git)
cd euphemism-detector

pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000`

---

## API Reference

### `POST /predict`

Classify whether a phrase is used euphemistically or literally within a sentence.

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "My grandfather passed away last Tuesday.",
    "phrase": "passed away"
  }'
```

**Response:**

```json
{
  "label": "Euphemistic",
  "conf_euphemistic": 94.2,
  "conf_literal": 5.8,
  "marked_input": "My grandfather [PET_BOUNDARY]passed away[PET_BOUNDARY] last Tuesday."
}
```

**Error — phrase not found:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sentence": "The sky is blue.", "phrase": "red"}'
```

```json
{"error": "Phrase not found in sentence"}
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

---

## Project Structure

```text
euphemism-detector/
├── .github/workflows/ci.yml   # CI: lint (ruff) + test (pytest) on Python 3.12/3.13
├── app.py                     # FastAPI application
├── static/                    # Frontend (HTML/CSS/JS)
├── tests/test_core.py         # Endpoint, validation, and model loading tests
├── Dockerfile                 # Container config for HuggingFace Spaces
├── requirements.txt           # Production dependencies
├── v1-streamlit/              # Prototype — English only, Streamlit UI
├── v3-multilingual/           # Multilingual training scripts
└── .github/workflows/         # CI (lint + test) and CD (deploy to HF Spaces)
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | `xlm-roberta-base` |
| Architecture | `XLMRobertaForSequenceClassification` |
| Labels | 0 = literal, 1 = euphemistic |
| Special tokens | `[PET_BOUNDARY]` (vocab size 250,003) |
| Input format | Sentence with `[PET_BOUNDARY]phrase[PET_BOUNDARY]` markers |
| Max length | 256 tokens |
| Training | AdamW, linear warmup, early stopping, stratified splits |

> **Note on Multilingual Performance:** While the architecture utilizes XLM-RoBERTa (which supports 100 languages), the current fine-tuned weights are primarily optimized for **English and Turkish**. Cross-lingual transfer capabilities for Spanish, Chinese, and Yoruba are currently under active evaluation and may default to literal predictions in zero-shot contexts.

### English Evaluation (v1)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Literal | 0.81 | 0.83 | 0.82 |
| Euphemistic | 0.88 | 0.86 | 0.87 |
| **Macro avg** | **0.84** | **0.84** | **0.84** |

---

## Development

### CD

Every push to `main` that passes CI is automatically deployed to HuggingFace Spaces.

### Run tests

```bash
pip install pytest httpx ruff
pytest tests/ -v
```

Tests use mocked models — no GPU or model weights required.

### Lint

```bash
ruff check . --select E,F --ignore E501
```

### CI

GitHub Actions runs linting and tests on every push/PR to `main` across Python 3.12 and 3.13.

---

## Roadmap

- [x] English fine-tuning — 84% macro F1
- [x] Custom web UI (FastAPI + HTML/CSS/JS)
- [x] Multilingual training — EN / TR / ES / ZH / YO
- [x] Docker containerization
- [x] CI pipeline (ruff + pytest)
- [x] CD pipeline — auto-deploy to HuggingFace Spaces on push
- [x] Auto-download model from HuggingFace Hub
- [ ] Multilingual evaluation results
- [ ] HuggingFace Hub model card

---

## Research Context

This project was developed as part of broader NLP research focusing on cross-lingual euphemism detection and transfer learning between Turkish and English. The underlying research and methodology associated with this repository have been accepted for presentation at the **SIGTURK 2026 workshop, co-located with EACL 2026**. 

Read the full preprint on arXiv: **[https://arxiv.org/abs/2602.16957](https://arxiv.org/abs/2602.16957)**

---

## License

MIT
EOF
```

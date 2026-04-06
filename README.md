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

**Live demo:** https://huggingface.co/spaces/hasancanbiyik/euphemism-detector

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
```

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
git clone https://github.com/hasancanbiyik/euphemism-detector.git
cd euphemism-detector

pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000

---

## API Reference

### `POST /predict`

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

---

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
├── .github/workflows/ci.yml
├── app.py
├── static/
├── tests/test_core.py
├── Dockerfile
├── requirements.txt
├── v1-streamlit/
├── v3-multilingual/
└── .github/workflows/
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | xlm-roberta-base |
| Architecture | XLMRobertaForSequenceClassification |
| Labels | 0 = literal, 1 = euphemistic |
| Special tokens | [PET_BOUNDARY] |
| Input format | Sentence with markers |
| Max length | 256 tokens |

---

## Development

```bash
pip install pytest httpx ruff
pytest tests/ -v
```

```bash
ruff check . --select E,F --ignore E501
```

---

## Roadmap

- [x] English fine-tuning
- [x] Multilingual training
- [x] Docker
- [x] CI/CD
- [ ] Multilingual evaluation

---

## Research Context

Accepted at SIGTURK 2026 (EACL 2026).

Preprint: https://arxiv.org/abs/2602.16957

---

## License

MIT

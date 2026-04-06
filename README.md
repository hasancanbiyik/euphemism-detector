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

### POST /predict

Classify whether a phrase is used euphemistically or literally within a sentence.

### GET /health

Returns model status.

---

## License

MIT

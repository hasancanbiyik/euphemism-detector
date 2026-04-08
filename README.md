---
title: Euphemism Detector
emoji: 🔍
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
---

# Euphemism Detector V2

[![CI](https://github.com/hasancanbiyik/euphemism-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/hasancanbiyik/euphemism-detector/actions/workflows/ci.yml)

A multilingual euphemism detection system powered by fine-tuned XLM-RoBERTa. Given a sentence and a target phrase, the model predicts whether the phrase is used **euphemistically or literally** in context, returning confidence scores for both classes. Trained on 7 languages, with preliminary zero-shot transfer observed across 22 additional languages spanning 12 language families.

**Live demo:** [HuggingFace Spaces](https://huggingface.co/spaces/hasancanbiyik/euphemism-detector)  
**Model weights:** [HuggingFace Hub](https://huggingface.co/hasancanbiyik/euphemism-detector-multilingual)  
**Paper (EACL 2026):** [arXiv:2602.16957](https://arxiv.org/abs/2602.16957)

---

## Why This Exists

Euphemisms are context-dependent — *"passed away"* signals death in one sentence and means nothing unusual in another. Standard sentiment or toxicity classifiers miss this because euphemism is a pragmatic phenomenon, not a lexical one. This project trains a cross-lingual transformer to detect euphemistic usage by learning contextual signals around **Potentially Euphemistic Terms (PETs)**.

Built as part of ongoing NLP research on cross-lingual euphemism detection and transfer learning. The V2 model extends the original 5-language system to 7 training languages with rigorous per-language evaluation, behavioral testing, and a batch prediction API.

---

## Performance

### Training Languages (19,490 examples)

| Language | Examples | Macro-F1 |
|----------|----------|----------|
| English | 3,098 | 0.800 |
| Turkish | 2,436 | 0.760 |
| Chinese (Mandarin) | 3,211 | 0.834 |
| Spanish | 2,952 | 0.828 |
| Yoruba | 2,598 | 0.840 |
| Polish | 2,439 | 0.810 |
| Ukrainian | 2,776 | 0.777 |
| **Overall** | **19,490** | **0.808** |

### Zero-Shot Cross-Lingual Transfer (22 unseen languages)

Preliminary evaluation using synthetically generated minimal pairs across 22 languages not present in training data. 15/22 languages exceeded 0.70 macro-F1, with 7 exceeding 0.85. Transfer strength correlates more with euphemistic semantic category than language family. See [Limitations](#limitations) for caveats on synthetic evaluation data.

| Language | Family | F1 | | Language | Family | F1 |
|----------|--------|-----|-|----------|--------|-----|
| Portuguese | Romance | 0.906 | | Korean | Koreanic | 0.804 |
| Indonesian | Austronesian | 0.899 | | Hungarian | Uralic | 0.800 |
| Swedish | Germanic | 0.899 | | Romanian | Romance | 0.800 |
| Hebrew | Semitic | 0.899 | | Arabic | Semitic | 0.792 |
| Danish | Germanic | 0.883 | | French | Romance | 0.708 |
| Hindi | Indo-Aryan | 0.862 | | Dutch | Germanic | 0.697 |
| German | Germanic | 0.844 | | Vietnamese | Austroasiatic | 0.697 |
| Italian | Romance | 0.829 | | Japanese | Japonic | 0.694 |

**Per-category transfer (zero-shot):** Appearance (F1: 1.00) > Bodily functions (0.82) > Death (0.79) > Employment (0.68)

### Behavioral Test Suite

26 tests covering negation robustness, cross-lingual consistency, confidence calibration, boundary token edge cases, and surface invariance. Result: **23 passed, 3 xfail** (documented limitations — negation sensitivity and rare PETs).

---

## Architecture

```text
┌─────────────────────────────────────────────────┐
│                   Client                        │
│          (Browser / curl / CSV upload)          │
└────────────────────┬────────────────────────────┘
                     │ HTTP
                     ▼
┌─────────────────────────────────────────────────┐
│             FastAPI Application                 │
│                                                 │
│  POST /predict                                  │
│    ├── Input validation + language detection    │
│    ├── [PET_BOUNDARY] marking around phrase     │
│    ├── XLMRobertaTokenizer encoding             │
│    └── Softmax → label + confidence scores      │
│                                                 │
│  POST /batch/predict                            │
│    ├── CSV upload (sentence, phrase columns)    │
│    ├── Batch inference (up to 10K rows)         │
│    └── JSON or CSV response with results       │
│                                                 │
│  GET /batch/template                            │
│    └── Download template CSV                    │
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
│  Training: 19.5K examples, 7 languages, fp16    │
└─────────────────────────────────────────────────┘
```

---

## Quickstart

### Run with Docker

```bash
docker build -t euphemism-detector .
docker run -p 7860:7860 euphemism-detector
```

Model weights are automatically downloaded from [HuggingFace Hub](https://huggingface.co/hasancanbiyik/euphemism-detector-multilingual) on first startup.

### Run locally

```bash
git clone https://github.com/hasancanbiyik/euphemism-detector.git
cd euphemism-detector
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000`

---

## API Reference

### `POST /predict`

Classify whether a phrase is used euphemistically or literally within a sentence.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "My grandfather passed away last Tuesday.",
    "phrase": "passed away"
  }'
```

```json
{
  "label": "Euphemistic",
  "conf_euphemistic": 94.2,
  "conf_literal": 5.8,
  "marked_input": "My grandfather [PET_BOUNDARY]passed away[PET_BOUNDARY] last Tuesday."
}
```

### `POST /batch/predict`

Upload a CSV with `sentence` and `phrase` columns for batch prediction.

```bash
curl -X POST http://localhost:8000/batch/predict \
  -F "file=@examples.csv"
```

Returns JSON array of predictions. Add `?format=csv` for CSV response.

### `GET /batch/template`

Download a template CSV for batch prediction.

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

---

## Project Structure

```text
euphemism-detector/
├── .github/workflows/
│   ├── ci.yml                 # CI: lint (ruff) + test (pytest) on Python 3.12/3.13
│   └── deploy.yml             # CD: auto-deploy to HuggingFace Spaces
├── static/                    # Frontend (HTML/CSS/JS) with batch QA UI
├── tests/
│   ├── test_core.py           # Unit tests (mocked model, no GPU needed)
│   └── test_behavioral.py     # Behavioral tests (26 tests, requires model)
├── app.py                     # FastAPI application
├── batch.py                   # Batch prediction router
├── Dockerfile                 # Container config for HuggingFace Spaces
├── requirements.txt           # Production dependencies
├── v1-streamlit/              # V1 prototype — English only, Streamlit UI
├── v2-fastapi/                # V2 FastAPI migration
└── v3-multilingual/           # Multilingual training scripts
```

---

## Model Details

| Property | Value |
|----------|-------|
| Base model | `xlm-roberta-base` |
| Architecture | `XLMRobertaForSequenceClassification` |
| Training languages | EN, TR, ZH, ES, YO, PL, UK |
| Training examples | 19,490 (stratified 80/10/10 split) |
| Labels | 0 = literal, 1 = euphemistic |
| Special tokens | `[PET_BOUNDARY]` (vocab size 250,003) |
| Input format | Sentence with `[PET_BOUNDARY]phrase[PET_BOUNDARY]` markers |
| Max length | 256 tokens |
| Training | AdamW, lr=1e-5, batch=32, fp16, class-weighted loss, early stopping (patience=5) |
| Overall macro-F1 | 0.808 |

---

## Development

### CI/CD

Every push to `main` that passes CI is automatically deployed to HuggingFace Spaces via GitHub Actions.

### Run tests

```bash
# Unit tests (no model needed)
pytest tests/test_core.py -v

# Behavioral tests (requires trained model)
MODEL_PATH=path/to/model pytest tests/test_behavioral.py -v
```

### Lint

```bash
ruff check . --select E,F --ignore E501
```

---

## Limitations

- **Negation sensitivity:** The model classifies negated euphemisms as literal (e.g., *"He didn't pass away"*). Negation context overwhelms the euphemistic signal. Documented as future work for adversarial training.
- **Zero-shot evaluation is synthetic:** The 22-language cross-lingual benchmark uses LLM-generated examples, not native-speaker-validated data. Distributional overlap with XLM-R's pretraining may inflate results. Native-speaker validation is needed before deployment claims for unseen languages.
- **Small zero-shot sample sizes:** 9–14 examples per unseen language. Results are preliminary.
- **Low-frequency PETs:** Rare or archaic euphemisms (e.g., *"powder her nose"*) may be misclassified.
- **Culture-specific euphemisms:** Best performance on universal categories (death, appearance). Culture-specific euphemisms without cross-lingual parallels may underperform.

---

## Roadmap

- [x] English fine-tuning — 84% macro F1 (V1)
- [x] Custom web UI (FastAPI + HTML/CSS/JS)
- [x] Multilingual training — 7 languages (EN/TR/ZH/ES/YO/PL/UK)
- [x] Docker containerization
- [x] CI pipeline (ruff + pytest)
- [x] CD pipeline — auto-deploy to HuggingFace Spaces
- [x] Auto-download model from HuggingFace Hub
- [x] Behavioral test suite (26 tests — negation, invariance, cross-lingual)
- [x] Batch QA feature (CSV upload → batch predictions)
- [x] Per-language evaluation results (0.808 overall macro-F1)
- [x] Zero-shot evaluation across 22 unseen languages (12 language families)
- [x] HuggingFace Hub model card
- [ ] Native-speaker validation of zero-shot benchmark (feel free to volunteer please!)
- [ ] SHAP/attention explainability dashboard
- [ ] Negation-aware adversarial training
- [ ] Browser extension for real-time euphemism detection
- [ ] REST API with authentication for third-party integrations

---

## Research Context

This project was developed as part of broader NLP research on cross-lingual euphemism detection:

- **Biyik, H. C.**, Barak, L., Peng, J., & Feldman, A. (2026). *When Semantic Overlap Is Not Enough: Cross-Lingual Euphemism Transfer Between Turkish and English.* SIGTURK at EACL 2026. [arXiv:2602.16957](https://arxiv.org/abs/2602.16957)

- **Biyik, H. C.**, Lee, P., & Feldman, A. (2024). *Turkish Delights: A Dataset on Turkish Euphemisms.* SIGTURK at ACL 2024. [arXiv:2407.13040](https://arxiv.org/abs/2407.13040)

- Lee, P., et al. (2024). *MEDs for PETs: Multilingual Euphemism Disambiguation for Potentially Euphemistic Terms.* Findings of EACL 2024.

The zero-shot cross-lingual evaluation extends the future work called for in Lee et al. (2024): *"More work is needed to test other training parameters and languages from a variety of families."*

---

## License

MIT

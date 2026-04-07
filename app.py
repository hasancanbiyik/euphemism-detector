"""
app.py — FastAPI backend for Euphemism Detector
Run: uvicorn app:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
from batch import batch_router, init_batch
import re
import os
import logging
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Euphemism Detector",
    description="Multilingual euphemism detection powered by fine-tuned XLM-RoBERTa (7 languages)",
    version="2.0.0",
)

MODEL_PATH = os.environ.get("MODEL_PATH", "./model")
HF_REPO = os.environ.get("HF_REPO", "hasancanbiyik/euphemism-detector-multilingual")

tokenizer, model = None, None


def load_model():
    """Load model from local path, falling back to HuggingFace Hub download."""
    global tokenizer, model

    if not os.path.exists(MODEL_PATH):
        logger.info(f"Local model not found at {MODEL_PATH}, downloading from {HF_REPO}...")
        try:
            snapshot_download(repo_id=HF_REPO, local_dir=MODEL_PATH)
            logger.info("Model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    logger.info("Model loaded and ready.")
    return True


load_model()

if model is not None:
    init_batch(model, tokenizer)
app.include_router(batch_router)


class PredictRequest(BaseModel):
    sentence: str
    phrase: str


from langdetect import detect  # make sure this import exists

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Model not loaded"}

    sentence = req.sentence.strip()
    phrase = req.phrase.strip()

    if phrase.lower() not in sentence.lower():
        return {"error": "Phrase not found in sentence"}

    marked = re.sub(
        re.escape(phrase),
        f"[PET_BOUNDARY]{phrase}[PET_BOUNDARY]",
        sentence,
        count=1,
        flags=re.IGNORECASE,
    )

    inputs = tokenizer(
        marked,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    conf_literal = round(probs[0].item() * 100, 1)
    conf_euph = round(probs[1].item() * 100, 1)
    label = "Euphemistic" if conf_euph > conf_literal else "Literal"

    # --- NEW: Language detection ---
    try:
        lang_code = detect(sentence)
        lang_map = {
            'en': 'English',
            'tr': 'Turkish',
            'es': 'Spanish',
            'zh-cn': 'Chinese',
            'zh-tw': 'Chinese',
            'yo': 'Yoruba',
            'pl': 'Polish',
            'uk': 'Ukrainian'
        }
        detected_lang = lang_map.get(lang_code, lang_code.upper())
    except:
        detected_lang = "Unknown"
    # --------------------------------

    return {
        "label": label,
        "conf_euphemistic": conf_euph,
        "conf_literal": conf_literal,
        "marked_input": marked,
        "language": detected_lang  # <-- added here
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# Serve static frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

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
import re
import os

app = FastAPI()

MODEL_PATH = "./model"
tokenizer, model = None, None

def load_model():
    global tokenizer, model
    if not os.path.exists(MODEL_PATH):
        return False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return True

load_model()

class PredictRequest(BaseModel):
    sentence: str
    phrase: str

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

    inputs = tokenizer(marked, return_tensors="pt", max_length=256, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    conf_literal = round(probs[0].item() * 100, 1)
    conf_euph    = round(probs[1].item() * 100, 1)
    label = "Euphemistic" if conf_euph > conf_literal else "Literal"

    return {
        "label": label,
        "conf_euphemistic": conf_euph,
        "conf_literal": conf_literal,
        "marked_input": marked,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# Serve static frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

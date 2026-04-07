"""
batch.py — Batch prediction router for Euphemism Detector V2
=============================================================
Add to your app.py:  from batch import batch_router; app.include_router(batch_router)

Provides:
  POST /batch/predict — Upload a CSV, get batch predictions back as JSON or CSV
  GET  /batch/template — Download a template CSV

CSV format expected:
  sentence,phrase
  "My grandmother passed away last Tuesday.","passed away"
  "The ball passed away from the goalkeeper.","passed away"
"""

import csv
import io
import re
from typing import List

import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

batch_router = APIRouter(prefix="/batch", tags=["batch"])

# These get set by app.py after model loading
_model = None
_tokenizer = None
BOUNDARY = "[PET_BOUNDARY]"


def init_batch(model, tokenizer):
    """Called from app.py to inject model references."""
    global _model, _tokenizer
    _model = model
    _tokenizer = tokenizer


class BatchResult(BaseModel):
    sentence: str
    phrase: str
    label: str
    conf_euphemistic: float
    conf_literal: float
    marked_input: str
    error: str = ""


def predict_single(sentence: str, phrase: str) -> dict:
    """Predict for a single sentence/phrase pair."""
    sentence = sentence.strip()
    phrase = phrase.strip()

    if not sentence or not phrase:
        return {"error": "Empty sentence or phrase"}

    if phrase.lower() not in sentence.lower():
        return {"error": f"Phrase '{phrase}' not found in sentence"}

    marked = re.sub(
        re.escape(phrase),
        f"{BOUNDARY}{phrase}{BOUNDARY}",
        sentence,
        count=1,
        flags=re.IGNORECASE,
    )

    inputs = _tokenizer(marked, return_tensors="pt", max_length=256, truncation=True, padding=True)
    with torch.no_grad():
        outputs = _model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    conf_literal = round(probs[0].item() * 100, 1)
    conf_euph = round(probs[1].item() * 100, 1)
    label = "Euphemistic" if conf_euph > conf_literal else "Literal"

    return {
        "label": label,
        "conf_euphemistic": conf_euph,
        "conf_literal": conf_literal,
        "marked_input": marked,
        "error": "",
    }


@batch_router.post("/predict", response_model=List[BatchResult])
async def batch_predict(file: UploadFile = File(...), format: str = "json"):
    """
    Upload a CSV with 'sentence' and 'phrase' columns.
    Returns batch predictions as JSON (default) or CSV.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and validate CSV
    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))

        if not reader.fieldnames:
            raise HTTPException(status_code=400, detail="Empty CSV file")

        # Normalize column names (case-insensitive, strip whitespace)
        normalized = {col.strip().lower(): col for col in reader.fieldnames}
        if "sentence" not in normalized or "phrase" not in normalized:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must have 'sentence' and 'phrase' columns. Found: {reader.fieldnames}",
            )

        sent_col = normalized["sentence"]
        phrase_col = normalized["phrase"]

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded CSV")

    # Process each row
    results = []
    reader = csv.DictReader(io.StringIO(text))  # re-read

    for i, row in enumerate(reader):
        if i >= 10000:  # safety cap
            break

        sentence = row.get(sent_col, "").strip()
        phrase = row.get(phrase_col, "").strip()

        pred = predict_single(sentence, phrase)
        results.append(BatchResult(
            sentence=sentence,
            phrase=phrase,
            label=pred.get("label", ""),
            conf_euphemistic=pred.get("conf_euphemistic", 0),
            conf_literal=pred.get("conf_literal", 0),
            marked_input=pred.get("marked_input", ""),
            error=pred.get("error", ""),
        ))

    # Return as CSV if requested
    if format.lower() == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["sentence", "phrase", "label", "conf_euphemistic", "conf_literal", "marked_input", "error"])
        for r in results:
            writer.writerow([r.sentence, r.phrase, r.label, r.conf_euphemistic, r.conf_literal, r.marked_input, r.error])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=batch_results.csv"},
        )

    return results


@batch_router.get("/template")
async def download_template():
    """Download a template CSV for batch prediction."""
    template = """sentence,phrase
"My grandmother passed away last Tuesday.","passed away"
"The ball passed away from the goalkeeper into the net.","passed away"
"He was let go from the company after the restructuring.","let go"
"She let go of the balloon and it floated into the sky.","let go"
"She is expecting a baby in the spring.","expecting"
"""
    return StreamingResponse(
        iter([template]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_template.csv"},
    )

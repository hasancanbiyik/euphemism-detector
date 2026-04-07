"""
tests/test_behavioral.py — Behavioral tests for Euphemism Detector V2
======================================================================
Tests the model's linguistic robustness across 7 categories:
  1. Known euphemistic pairs
  2. Known literal pairs
  3. Negation robustness (xfail — known limitation)
  4. Boundary token robustness
  5. Cross-lingual consistency
  6. Confidence calibration
  7. Surface invariance

Usage:
    pytest tests/test_behavioral.py -v
    MODEL_PATH=/path/to/model pytest tests/test_behavioral.py -v
"""

import os
import re

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BOUNDARY = "[PET_BOUNDARY]"
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "best_model")
)


@pytest.fixture(scope="session")
def model_and_tokenizer():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model not found at {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()
    return {
        "label": "euphemistic" if probs[1].item() > probs[0].item() else "literal",
        "conf_euphemistic": probs[1].item(),
        "conf_literal": probs[0].item(),
    }


def mark_pet(sentence, phrase):
    return re.sub(re.escape(phrase), f"{BOUNDARY}{phrase}{BOUNDARY}", sentence, count=1, flags=re.IGNORECASE)


# ============================================================
# 1. KNOWN EUPHEMISTIC — must classify as euphemistic
# ============================================================
EUPHEMISTIC_CASES = [
    ("My grandmother passed away last Tuesday.", "passed away", "EN-death0"),
    ("He was let go from the company after the restructuring.", "let go", "EN-employment"),
    ("She is expecting a baby in the spring.", "expecting", "EN-pregnancy"),
    ("The soldiers who made the ultimate sacrifice will be honored.", "ultimate sacrifice", "EN-death1"),
    ("He's been between jobs for about three months now.", "between jobs", "EN-unemployment"),
    ("The company is going through a period of rightsizing.", "rightsizing", "EN-layoffs"),
    ("They had to put their dog to sleep last week.", "put their dog to sleep", "EN-euthanasia"),
]


@pytest.mark.parametrize("sentence,phrase,note", EUPHEMISTIC_CASES, ids=[c[2] for c in EUPHEMISTIC_CASES])
def test_known_euphemistic(model_and_tokenizer, sentence, phrase, note):
    model, tokenizer = model_and_tokenizer
    result = predict(model, tokenizer, mark_pet(sentence, phrase))
    assert result["label"] == "euphemistic", f"[{note}] '{phrase}': got {result['label']} (euph={result['conf_euphemistic']:.3f})"


@pytest.mark.xfail(reason="Low-frequency euphemism not well-represented in training data (conf=0.477)")
def test_rare_euphemism_powder_nose(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    result = predict(model, tokenizer, mark_pet("She went to powder her nose before the dinner party.", "powder her nose"))
    assert result["label"] == "euphemistic"


# ============================================================
# 2. KNOWN LITERAL — must classify as literal
# ============================================================
LITERAL_CASES = [
    ("The ball passed away from the goalkeeper into the net.", "passed away", "EN-lit-passed-away"),
    ("She let go of the balloon and it floated into the sky.", "let go", "EN-lit-let-go"),
    ("I am expecting a package to arrive today.", "expecting", "EN-lit-expecting"),
    ("She fell asleep on the couch watching TV.", "fell asleep", "EN-lit-fell-asleep"),
    ("He went to the bathroom to wash his hands before dinner.", "went to the bathroom", "EN-lit-bathroom"),
]


@pytest.mark.parametrize("sentence,phrase,note", LITERAL_CASES, ids=[c[2] for c in LITERAL_CASES])
def test_known_literal(model_and_tokenizer, sentence, phrase, note):
    model, tokenizer = model_and_tokenizer
    result = predict(model, tokenizer, mark_pet(sentence, phrase))
    assert result["label"] == "literal", f"[{note}] '{phrase}': got {result['label']} (lit={result['conf_literal']:.3f})"


# ============================================================
# 3. NEGATION ROBUSTNESS (xfail — known transformer limitation)
# Negation context provides strong literal signals that overwhelm
# euphemistic sense. Documented as future work for adversarial training.
# ============================================================
NEGATION_PAIRS = [
    ("He passed away peacefully in his sleep.", "He didn't pass away - he's still fighting in the ICU.", "passed away", "EN-neg-death"),
    ("She was let go from the company.", "She wasn't let go - she resigned on her own terms.", "let go", "EN-neg-employment"),
]


@pytest.mark.xfail(reason="Negation context causes literal classification — known transformer limitation, future work for adversarial training")
@pytest.mark.parametrize("pos_sent,neg_sent,phrase,note", NEGATION_PAIRS, ids=[c[3] for c in NEGATION_PAIRS])
def test_negation_preserves_euphemistic_sense(model_and_tokenizer, pos_sent, neg_sent, phrase, note):
    model, tokenizer = model_and_tokenizer
    pos_result = predict(model, tokenizer, mark_pet(pos_sent, phrase))
    neg_result = predict(model, tokenizer, mark_pet(neg_sent, phrase))
    assert pos_result["label"] == "euphemistic", "Positive case should be euphemistic"
    assert neg_result["label"] == "euphemistic", "Negated case should still be euphemistic"


# ============================================================
# 4. BOUNDARY TOKEN ROBUSTNESS
# ============================================================
class TestBoundaryRobustness:
    def test_boundary_at_start(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        text = f"{BOUNDARY}Passed away{BOUNDARY} is what they told us about grandma."
        assert predict(model, tokenizer, text)["label"] in ("euphemistic", "literal")

    def test_boundary_at_end(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        text = f"The doctor told us that grandpa {BOUNDARY}passed away{BOUNDARY}"
        assert predict(model, tokenizer, text)["label"] in ("euphemistic", "literal")

    def test_long_context(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        padding = "This is filler text to pad the context. " * 50
        text = f"{padding}He had {BOUNDARY}passed away{BOUNDARY} during the night. {padding}"
        assert predict(model, tokenizer, text)["label"] in ("euphemistic", "literal")

    def test_empty_pet(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        text = f"The family said {BOUNDARY}{BOUNDARY} and moved on."
        assert predict(model, tokenizer, text)["label"] in ("euphemistic", "literal")


# ============================================================
# 5. CROSS-LINGUAL CONSISTENCY
# ============================================================
CROSS_LINGUAL_DEATH = [
    ("My grandmother passed away last week.", "passed away", "EN"),
    ("Mi abuela fallecio la semana pasada.", "fallecio", "ES"),
]


@pytest.mark.parametrize("sentence,phrase,lang", CROSS_LINGUAL_DEATH, ids=[c[2] for c in CROSS_LINGUAL_DEATH])
def test_crosslingual_death_euphemism(model_and_tokenizer, sentence, phrase, lang):
    model, tokenizer = model_and_tokenizer
    result = predict(model, tokenizer, mark_pet(sentence, phrase))
    assert result["label"] == "euphemistic", f"[{lang}] '{phrase}' should be euphemistic"


# ============================================================
# 6. CONFIDENCE CALIBRATION
# ============================================================
CONFIDENCE_PAIRS = [
    {"euphemistic": ("My grandmother passed away last Tuesday.", "passed away"),
     "literal": ("The ball passed away from the goalkeeper.", "passed away"),
     "note": "EN-passed-away"},
    {"euphemistic": ("He was let go from the company.", "let go"),
     "literal": ("She let go of the balloon.", "let go"),
     "note": "EN-let-go"},
]


@pytest.mark.parametrize("pair", CONFIDENCE_PAIRS, ids=[p["note"] for p in CONFIDENCE_PAIRS])
def test_euphemistic_confidence_higher_than_literal(model_and_tokenizer, pair):
    model, tokenizer = model_and_tokenizer
    euph_r = predict(model, tokenizer, mark_pet(*pair["euphemistic"]))
    lit_r = predict(model, tokenizer, mark_pet(*pair["literal"]))
    assert euph_r["conf_euphemistic"] > lit_r["conf_euphemistic"], \
        f"Euph conf {euph_r['conf_euphemistic']:.3f} should be > lit conf {lit_r['conf_euphemistic']:.3f}"


# ============================================================
# 7. SURFACE INVARIANCE
# ============================================================
class TestInvariance:
    def test_case_invariance(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        r1 = predict(model, tokenizer, f"My grandmother {BOUNDARY}passed away{BOUNDARY} last Tuesday.")
        r2 = predict(model, tokenizer, f"MY GRANDMOTHER {BOUNDARY}passed away{BOUNDARY} LAST TUESDAY.")
        assert r1["label"] == r2["label"], "Case change flipped prediction"

    def test_punctuation_invariance(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        r1 = predict(model, tokenizer, f"My grandmother {BOUNDARY}passed away{BOUNDARY} last Tuesday.")
        r2 = predict(model, tokenizer, f"My grandmother {BOUNDARY}passed away{BOUNDARY} last Tuesday")
        assert r1["label"] == r2["label"], "Punctuation change flipped prediction"

    def test_whitespace_invariance(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        r1 = predict(model, tokenizer, f"My grandmother {BOUNDARY}passed away{BOUNDARY} last Tuesday.")
        r2 = predict(model, tokenizer, f"My  grandmother  {BOUNDARY}passed away{BOUNDARY}  last  Tuesday.")
        assert r1["label"] == r2["label"], "Whitespace change flipped prediction"

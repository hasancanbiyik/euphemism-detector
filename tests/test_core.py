"""
tests/test_core.py — Unit tests for Euphemism Detector API

Tests run without a GPU or model weights on disk.
Model-dependent tests are skipped when ./model is absent.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client_no_model():
    """Client with model explicitly set to None (simulates missing weights)."""
    import app as app_module

    app_module.model = None
    app_module.tokenizer = None
    return TestClient(app_module.app)


@pytest.fixture()
def client_mock_model():
    """Client with a mocked model that returns deterministic logits."""
    import app as app_module

    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }

    mock_outputs = MagicMock()
    # logits: literal=0.2, euphemistic=0.8 → model predicts euphemistic
    mock_outputs.logits = torch.tensor([[0.2, 0.8]])

    mock_model = MagicMock()
    mock_model.return_value = mock_outputs
    mock_model.eval = MagicMock()

    app_module.tokenizer = mock_tokenizer
    app_module.model = mock_model

    return TestClient(app_module.app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_reports_model_not_loaded(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.json()["model_loaded"] is False

    def test_health_reports_model_loaded(self, client_mock_model):
        resp = client_mock_model.get("/health")
        assert resp.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# /predict — validation
# ---------------------------------------------------------------------------

class TestPredictValidation:
    def test_missing_body_returns_422(self, client_mock_model):
        resp = client_mock_model.post("/predict")
        assert resp.status_code == 422

    def test_missing_phrase_field_returns_422(self, client_mock_model):
        resp = client_mock_model.post("/predict", json={"sentence": "Hello world"})
        assert resp.status_code == 422

    def test_missing_sentence_field_returns_422(self, client_mock_model):
        resp = client_mock_model.post("/predict", json={"phrase": "hello"})
        assert resp.status_code == 422

    def test_phrase_not_in_sentence(self, client_mock_model):
        resp = client_mock_model.post(
            "/predict",
            json={"sentence": "The sky is blue.", "phrase": "red"},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()
        assert "not found" in resp.json()["error"].lower()

    def test_model_not_loaded_returns_error(self, client_no_model):
        resp = client_no_model.post(
            "/predict",
            json={"sentence": "He passed away.", "phrase": "passed away"},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()


# ---------------------------------------------------------------------------
# /predict — inference (mocked model)
# ---------------------------------------------------------------------------

class TestPredictInference:
    def test_euphemistic_prediction(self, client_mock_model):
        resp = client_mock_model.post(
            "/predict",
            json={
                "sentence": "My grandfather passed away last year.",
                "phrase": "passed away",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" not in data
        assert data["label"] == "Euphemistic"
        assert "conf_euphemistic" in data
        assert "conf_literal" in data
        assert data["conf_euphemistic"] > data["conf_literal"]

    def test_marked_input_contains_boundary_tokens(self, client_mock_model):
        resp = client_mock_model.post(
            "/predict",
            json={
                "sentence": "She was let go from the company.",
                "phrase": "let go",
            },
        )
        data = resp.json()
        assert "[PET_BOUNDARY]" in data["marked_input"]
        assert data["marked_input"].count("[PET_BOUNDARY]") == 2

    def test_case_insensitive_phrase_match(self, client_mock_model):
        resp = client_mock_model.post(
            "/predict",
            json={
                "sentence": "He Passed Away yesterday.",
                "phrase": "passed away",
            },
        )
        data = resp.json()
        assert "error" not in data
        assert data["label"] in ("Euphemistic", "Literal")

    def test_confidence_scores_sum_to_100(self, client_mock_model):
        resp = client_mock_model.post(
            "/predict",
            json={
                "sentence": "The company downsized its workforce.",
                "phrase": "downsized",
            },
        )
        data = resp.json()
        total = data["conf_euphemistic"] + data["conf_literal"]
        assert abs(total - 100.0) < 0.5  # allow small rounding error

    def test_response_schema(self, client_mock_model):
        resp = client_mock_model.post(
            "/predict",
            json={
                "sentence": "They swept it under the rug.",
                "phrase": "under the rug",
            },
        )
        data = resp.json()
        expected_keys = {"label", "conf_euphemistic", "conf_literal", "marked_input"}
        assert expected_keys == set(data.keys())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_load_model_downloads_when_path_missing(self):
        """When local path doesn't exist, snapshot_download is called."""
        import app as app_module

        mock_model = MagicMock()

        with (
            patch("os.path.exists", return_value=False),
            patch.object(app_module, "snapshot_download") as mock_download,
            patch.object(app_module, "AutoTokenizer") as mock_tok_cls,
            patch.object(app_module, "AutoModelForSequenceClassification") as mock_model_cls,
        ):
            mock_model_cls.from_pretrained.return_value = mock_model
            mock_tok_cls.from_pretrained.return_value = MagicMock()

            result = app_module.load_model()
            mock_download.assert_called_once()
            assert result is True

    def test_load_model_returns_false_when_download_fails(self):
        """When local path doesn't exist and download fails, returns False."""
        import app as app_module

        with (
            patch("os.path.exists", return_value=False),
            patch.object(app_module, "snapshot_download", side_effect=Exception("Network error")),
        ):
            result = app_module.load_model()
            assert result is False

    def test_load_model_skips_download_when_path_exists(self):
        """When local path exists, no download is attempted."""
        import app as app_module

        mock_model = MagicMock()

        with (
            patch("os.path.exists", return_value=True),
            patch.object(app_module, "snapshot_download") as mock_download,
            patch.object(app_module, "AutoTokenizer") as mock_tok_cls,
            patch.object(app_module, "AutoModelForSequenceClassification") as mock_model_cls,
        ):
            mock_tok_cls.from_pretrained.return_value = MagicMock()
            mock_model_cls.from_pretrained.return_value = mock_model

            result = app_module.load_model()
            mock_download.assert_not_called()
            assert result is True
            mock_model.eval.assert_called_once()

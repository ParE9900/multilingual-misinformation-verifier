"""Tests for verification pipeline."""

from unittest.mock import MagicMock, patch
import pytest
from src.config import Config
from src.pipeline import VerificationPipeline


@pytest.fixture
def mock_config(tmp_path):
    return Config(
        GROQ_API_KEY="mock_groq",
        GEMINI_API_KEY="mock_gemini",
        HF_TOKEN="mock_hf",
        CACHE_DIR=str(tmp_path / "cache.db"),
    )


def test_pipeline_verify_success(mock_config):
    with patch("src.pipeline.NLIClassifier") as mock_nli_cls, \
         patch("src.pipeline.ClaimDecomposer") as mock_dec_cls, \
         patch("src.pipeline.EvidenceRetriever") as mock_ret_cls:

        mock_nli = MagicMock()
        mock_nli.detect_language.return_value = "en"
        mock_nli.predict_stance.return_value = {"supports": 0.8, "refutes": 0.2}
        mock_nli_cls.return_value = mock_nli

        mock_dec = MagicMock()
        mock_dec.decompose.return_value = ["sub claim 1", "sub claim 2"]
        mock_dec_cls.return_value = mock_dec

        mock_ret = MagicMock()
        mock_ret.retrieve.return_value = [
            {"url": "http://example.com", "snippet": "example snippet", "title": "Example"}
        ]
        mock_ret_cls.return_value = mock_ret

        pipeline = VerificationPipeline(config=mock_config)
        res = pipeline.verify("A test claim that is mostly real.")

        assert res["status"] == "ok"
        assert "probability_fake" in res
        assert "verdict" in res
        assert "confidence_interval" in res
        assert "evidence_trace" in res
        assert "language" in res
        assert res["language"] == "en"
        assert "timings_ms" in res
        assert len(res["evidence_trace"]) == 2


def test_pipeline_one_subclaim_fails_continues(mock_config):
    with patch("src.pipeline.NLIClassifier") as mock_nli_cls, \
         patch("src.pipeline.ClaimDecomposer") as mock_dec_cls, \
         patch("src.pipeline.EvidenceRetriever") as mock_ret_cls:

        mock_nli = MagicMock()
        mock_nli.detect_language.return_value = "en"
        mock_nli.predict_stance.return_value = {"supports": 0.5, "refutes": 0.5}
        mock_nli_cls.return_value = mock_nli

        mock_dec = MagicMock()
        mock_dec.decompose.return_value = ["failing sub claim", "successful sub claim"]
        mock_dec_cls.return_value = mock_dec

        mock_ret = MagicMock()

        def side_effect(sc):
            if sc == "failing sub claim":
                raise RuntimeError("API timeout")
            return [{"url": "http://ok.com", "snippet": "evidence snippet", "title": "OK"}]

        mock_ret.retrieve.side_effect = side_effect
        mock_ret_cls.return_value = mock_ret

        pipeline = VerificationPipeline(config=mock_config)
        res = pipeline.verify("A mixed claim.")

        assert res["status"] == "ok"
        assert len(res["evidence_trace"]) == 1
        assert res["evidence_trace"][0]["sub_claim"] == "successful sub claim"


def test_pipeline_all_subclaims_fail_returns_error(mock_config):
    with patch("src.pipeline.NLIClassifier") as mock_nli_cls, \
         patch("src.pipeline.ClaimDecomposer") as mock_dec_cls, \
         patch("src.pipeline.EvidenceRetriever") as mock_ret_cls:

        mock_nli = MagicMock()
        mock_nli.detect_language.return_value = "en"
        mock_nli_cls.return_value = mock_nli

        mock_dec = MagicMock()
        mock_dec.decompose.return_value = ["sub1", "sub2"]
        mock_dec_cls.return_value = mock_dec

        mock_ret = MagicMock()
        mock_ret.retrieve.side_effect = RuntimeError("Service down")
        mock_ret_cls.return_value = mock_ret

        pipeline = VerificationPipeline(config=mock_config)
        res = pipeline.verify("All fail claim.")

        assert res["status"] == "error"
        assert "message" in res
        assert "failed" in res["message"].lower()

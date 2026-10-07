"""Unit tests for ClaimDecomposer module."""

from unittest.mock import MagicMock, patch
import pytest
from src.cache import DiskCache
from src.claim_decomposer import ClaimDecomposer


def test_decompose_success(tmp_path):
    cache = DiskCache(str(tmp_path / "cache.db"))
    with patch("src.claim_decomposer.ChatGroq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content='{"sub_claims": ["a", "b"]}')
        mock_groq_cls.return_value = mock_instance

        decomposer = ClaimDecomposer(api_key="test_key", cache=cache)
        sub_claims = decomposer.decompose("Sample compound claim", language="en")

        assert sub_claims == ["a", "b"]
        assert mock_instance.invoke.call_count == 1


def test_decompose_cache_hit(tmp_path):
    cache = DiskCache(str(tmp_path / "cache.db"))
    with patch("src.claim_decomposer.ChatGroq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content='{"sub_claims": ["a", "b"]}')
        mock_groq_cls.return_value = mock_instance

        decomposer = ClaimDecomposer(api_key="test_key", cache=cache)
        first_call = decomposer.decompose("Cached claim", language="en")
        second_call = decomposer.decompose("Cached claim", language="en")

        assert first_call == ["a", "b"]
        assert second_call == ["a", "b"]
        assert mock_instance.invoke.call_count == 1


def test_decompose_malformed_json_fallback(tmp_path):
    cache = DiskCache(str(tmp_path / "cache.db"))
    with patch("src.claim_decomposer.ChatGroq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="I am not valid JSON!")
        mock_groq_cls.return_value = mock_instance

        decomposer = ClaimDecomposer(api_key="test_key", cache=cache)
        original_claim = "Unparsable claim"
        sub_claims = decomposer.decompose(original_claim, language="en")

        assert sub_claims == [original_claim]

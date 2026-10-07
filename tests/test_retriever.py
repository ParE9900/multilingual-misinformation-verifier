"""Unit tests for EvidenceRetriever module."""

from unittest.mock import MagicMock, patch
import pytest
from src.cache import DiskCache
from src.evidence_retriever import EvidenceRetriever


def test_retrieve_with_grounding_chunk(tmp_path):
    cache = DiskCache(str(tmp_path / "cache.db"))
    with patch("src.evidence_retriever.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = {
            "grounding_chunks": [{"web": {"uri": "https://x.com", "title": "X"}}],
            "grounding_supports": [{"segment": {"text": "Snippet text"}, "grounding_chunk_indices": [0]}],
        }
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        retriever = EvidenceRetriever(api_key="test_key", cache=cache)
        results = retriever.retrieve("Test sub claim")

        assert len(results) == 1
        assert results[0]["url"] == "https://x.com"
        assert results[0]["title"] == "X"
        assert results[0]["snippet"] == "Snippet text"
        assert mock_client.models.generate_content.call_count == 1


def test_retrieve_without_grounding_chunks(tmp_path):
    cache = DiskCache(str(tmp_path / "cache.db"))
    with patch("src.evidence_retriever.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = {
            "grounding_chunks": []
        }
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        retriever = EvidenceRetriever(api_key="test_key", cache=cache)
        results = retriever.retrieve("Ungrounded sub claim")

        assert results == []
        assert mock_client.models.generate_content.call_count == 1


def test_retrieve_cache_hit(tmp_path):
    cache = DiskCache(str(tmp_path / "cache.db"))
    with patch("src.evidence_retriever.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_candidate = MagicMock()
        mock_candidate.grounding_metadata = {
            "grounding_chunks": [{"web": {"uri": "https://example.com", "title": "Example"}}]
        }
        mock_response.candidates = [mock_candidate]
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        retriever = EvidenceRetriever(api_key="test_key", cache=cache)
        first_call = retriever.retrieve("Repeat query")
        second_call = retriever.retrieve("Repeat query")

        assert first_call == [{"url": "https://example.com", "snippet": "", "title": "Example"}]
        assert second_call == first_call
        assert mock_client.models.generate_content.call_count == 1

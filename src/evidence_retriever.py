"""Evidence retrieval module using Gemini 3.5 Flash with grounded search."""

import time
from typing import Any, Dict, List, Optional
from google import genai
from src.cache import DiskCache


class EvidenceRetriever:
    """Retrieves grounded search evidence for factual sub-claims using Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash",
        cache: Optional[DiskCache] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.cache = cache
        self.client = genai.Client(api_key=self.api_key)

    def _call_gemini(self, sub_claim: str) -> Any:
        max_retries = 5
        base_delay = 2.0
        prompt = f"Find real-time evidence to verify or refute this claim: {sub_claim}"

        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"tools": [{"google_search": {}}]},
                )
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "503" in err_str or "resource_exhausted" in err_str or "unavailable" in err_str:
                    if attempt == max_retries - 1:
                        raise RuntimeError("Gemini API rate limit or service unavailable after retries.")
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    if attempt == max_retries - 1:
                        raise RuntimeError("Gemini API call failed after retries.")
                    time.sleep(base_delay * (2 ** attempt))

    def _extract_grounding(self, response: Any) -> List[Dict[str, str]]:
        if not response:
            return []

        grounding_metadata = None
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]
            if hasattr(cand, "grounding_metadata"):
                grounding_metadata = cand.grounding_metadata
            elif isinstance(cand, dict):
                grounding_metadata = cand.get("grounding_metadata")
        elif hasattr(response, "grounding_metadata"):
            grounding_metadata = response.grounding_metadata
        elif isinstance(response, dict):
            grounding_metadata = response.get("grounding_metadata")

        if not grounding_metadata:
            return []

        chunks = None
        if hasattr(grounding_metadata, "grounding_chunks"):
            chunks = grounding_metadata.grounding_chunks
        elif isinstance(grounding_metadata, dict):
            chunks = grounding_metadata.get("grounding_chunks")

        if not chunks:
            return []

        supports = None
        if hasattr(grounding_metadata, "grounding_supports"):
            supports = grounding_metadata.grounding_supports
        elif isinstance(grounding_metadata, dict):
            supports = grounding_metadata.get("grounding_supports")

        chunk_snippets = {}
        if supports:
            for support in supports:
                text = ""
                if hasattr(support, "segment"):
                    seg = support.segment
                    text = getattr(seg, "text", "") if hasattr(seg, "text") else (seg.get("text", "") if isinstance(seg, dict) else "")
                elif isinstance(support, dict) and "segment" in support:
                    seg = support["segment"]
                    text = seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")

                indices = []
                if hasattr(support, "grounding_chunk_indices"):
                    indices = support.grounding_chunk_indices or []
                elif isinstance(support, dict):
                    indices = support.get("grounding_chunk_indices", [])

                for idx in indices:
                    if idx not in chunk_snippets and text:
                        chunk_snippets[idx] = text

        results = []
        for i, chunk in enumerate(chunks):
            url = ""
            title = ""
            snippet = chunk_snippets.get(i, "")

            web = None
            if hasattr(chunk, "web"):
                web = chunk.web
            elif isinstance(chunk, dict):
                web = chunk.get("web")

            if web:
                if hasattr(web, "uri"):
                    url = web.uri or ""
                elif isinstance(web, dict):
                    url = web.get("uri", "")

                if hasattr(web, "title"):
                    title = web.title or ""
                elif isinstance(web, dict):
                    title = web.get("title", "")

            if not url:
                if hasattr(chunk, "uri"):
                    url = chunk.uri
                elif isinstance(chunk, dict):
                    url = chunk.get("uri", chunk.get("url", ""))

            if not title:
                if hasattr(chunk, "title"):
                    title = chunk.title
                elif isinstance(chunk, dict):
                    title = chunk.get("title", "")

            if not snippet:
                if hasattr(chunk, "snippet"):
                    snippet = chunk.snippet
                elif isinstance(chunk, dict):
                    snippet = chunk.get("snippet", "")

            results.append({
                "url": str(url) if url else "",
                "snippet": str(snippet) if snippet else "",
                "title": str(title) if title else "",
            })

            if len(results) >= 3:
                break

        return results

    def retrieve(self, sub_claim: str) -> List[Dict[str, str]]:
        sub_claim = sub_claim.strip()
        if not sub_claim:
            return []

        if self.cache is not None:
            cached = self.cache.get("evidence_retrieval", sub_claim)
            if cached is not None:
                return cached

        response = self._call_gemini(sub_claim)
        results = self._extract_grounding(response)

        if self.cache is not None and results is not None:
            self.cache.set("evidence_retrieval", sub_claim, results)

        return results

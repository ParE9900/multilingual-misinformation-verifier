"""Evidence retrieval module using Gemini with grounded search."""

import time
from typing import Any, Dict, List, Optional
from google import genai
from src.cache import DiskCache


class EvidenceRetriever:
    """Retrieves grounded search evidence for factual sub-claims."""

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

    def _call_gemini(self, sub_claim: str, use_search: bool = True) -> Any:
        max_retries = 2
        base_delay = 2.0
        prompt = f"Find real-time evidence to verify or refute this claim: {sub_claim}"
        
        config = {"tools": [{"google_search": {}}]} if use_search else None

        for attempt in range(max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "503" in err_str or "resource_exhausted" in err_str or "unavailable" in err_str:
                    if attempt == max_retries - 1:
                        raise RuntimeError("Gemini API rate limit or service unavailable.")
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    raise RuntimeError(f"Gemini API call failed: {e}")

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
        """Retrieve grounded evidence for sub-claim, with fallback to base knowledge."""
        sub_claim = sub_claim.strip()
        if not sub_claim:
            return []

        if self.cache is not None:
            cached = self.cache.get("evidence_retrieval", sub_claim)
            if cached is not None:
                return cached

        # 1. Try with Search Grounding
        try:
            response = self._call_gemini(sub_claim, use_search=True)
            results = self._extract_grounding(response)
            if results:
                if self.cache is not None:
                    self.cache.set("evidence_retrieval", sub_claim, results)
                return results
        except Exception as e:
            print(f"[DEBUG] Search grounding failed for '{sub_claim}': {e}. Falling back to base knowledge.")

        # 2. Fallback to base knowledge (no search tool)
        try:
            response = self._call_gemini(sub_claim, use_search=False)
            snippet = response.text if hasattr(response, "text") else str(response)
            fallback_result = [{
                "url": "", 
                "snippet": snippet, 
                "title": "Generated Context (Search Unavailable)"
            }]
            if self.cache is not None:
                self.cache.set("evidence_retrieval", sub_claim, fallback_result)
            return fallback_result
        except Exception as e:
            print(f"[DEBUG] Base Gemini generation failed for '{sub_claim}': {e}")
            return []
"""Claim decomposition module using Groq Llama 3.3."""

import json
import time
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from src.cache import DiskCache

SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Decompose the following claim into 3-7 atomic, "
    "independently verifiable sub-claims. Output strict JSON in the format: "
    '{"sub_claims": ["sub-claim 1", "sub-claim 2"]}. Do not include any other text or commentary.'
)
REPAIR_PROMPT = "Your previous response was not valid JSON. Please output ONLY valid JSON."


class ClaimDecomposer:
    """Decomposes compound claims into atomic verifiable sub-claims."""

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        cache: Optional[DiskCache] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.cache = cache
        self.client = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model,
            temperature=0.0,
        )

    def _call_groq(self, messages: list) -> str:
        max_retries = 5
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                response = self.client.invoke(messages)
                if hasattr(response, "content"):
                    return str(response.content)
                return str(response)
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate_limit" in err_str or "too many requests" in err_str:
                    if attempt == max_retries - 1:
                        raise RuntimeError("Groq API rate limit reached after retries.")
                    time.sleep(base_delay * (2 ** attempt))
                else:
                    if attempt == max_retries - 1:
                        raise RuntimeError("Groq API request failed after retries.")
                    time.sleep(base_delay * (2 ** attempt))
        return ""

    def _parse_json(self, raw_text: str) -> Optional[List[str]]:
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "sub_claims" in data and isinstance(data["sub_claims"], list):
                return [str(s) for s in data["sub_claims"] if str(s).strip()]
        except Exception:
            pass
        return None

    def decompose(self, claim: str, language: str = "en") -> List[str]:
        claim = claim.strip()
        if not claim:
            return []

        cache_key = f"{language}:{claim}"
        if self.cache is not None:
            cached = self.cache.get("claim_decomposition", cache_key)
            if cached is not None:
                return cached

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=claim),
        ]

        raw_response = self._call_groq(messages)
        parsed = self._parse_json(raw_response)

        if parsed is None:
            repair_messages = list(messages) + [
                HumanMessage(content=REPAIR_PROMPT)
            ]
            repair_response = self._call_groq(repair_messages)
            parsed = self._parse_json(repair_response)

        result = parsed if parsed else [claim]

        if self.cache is not None:
            self.cache.set("claim_decomposition", cache_key, result)

        return result

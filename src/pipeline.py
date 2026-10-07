"""End-to-end verification pipeline orchestrator."""

from pathlib import Path
import time
from typing import Any, Dict, List
from src.bayes_aggregator import aggregate
from src.cache import DiskCache
from src.claim_decomposer import ClaimDecomposer
from src.config import Config
from src.evidence_retriever import EvidenceRetriever
from src.nli_classifier import NLIClassifier


class VerificationPipeline:
    """Orchestrates language detection, decomposition, evidence retrieval, and Bayesian fusion."""

    def __init__(self, config: Config):
        cache_path = Path(config.CACHE_DIR)
        if cache_path.suffix != ".db":
            cache_path = cache_path / "cache.db"

        self.cache = DiskCache(path=str(cache_path))
        self.nli = NLIClassifier(model_name=config.NLI_MODEL_NAME)
        self.nli.load()

        self.decomposer = ClaimDecomposer(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            cache=self.cache,
        )

        self.retriever = EvidenceRetriever(
            api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            cache=self.cache,
        )

    def verify(self, claim: str) -> Dict[str, Any]:
        """Execute fact-checking pipeline on a user claim."""
        try:
            claim_str = (claim or "").strip()
            if not claim_str:
                return {"status": "error", "message": "Empty claim provided."}

            timings_ms: Dict[str, float] = {}

            t0 = time.perf_counter()
            try:
                language = self.nli.detect_language(claim_str)
            except Exception:
                language = "en"
            timings_ms["detect_language"] = round((time.perf_counter() - t0) * 1000, 2)

            t0 = time.perf_counter()
            try:
                sub_claims = self.decomposer.decompose(claim_str, language=language)
            except Exception:
                sub_claims = [claim_str]
            timings_ms["decompose"] = round((time.perf_counter() - t0) * 1000, 2)

            if not sub_claims:
                sub_claims = [claim_str]

            t0 = time.perf_counter()
            subclaim_results: List[Dict[str, Any]] = []

            for sub_claim in sub_claims:
                try:
                    sources = self.retriever.retrieve(sub_claim)
                    top_evidence = ""
                    if sources and isinstance(sources, list) and len(sources) > 0:
                        first_source = sources[0]
                        if isinstance(first_source, dict):
                            top_evidence = (
                                first_source.get("snippet", "")
                                or first_source.get("title", "")
                            )

                    stance = self.nli.predict_stance(claim=sub_claim, evidence=top_evidence)
                    subclaim_results.append({
                        "sub_claim": sub_claim,
                        "stance": stance,
                        "sources": sources or [],
                    })
                except Exception:
                    continue

            timings_ms["retrieve_and_classify"] = round((time.perf_counter() - t0) * 1000, 2)

            if not subclaim_results:
                return {
                    "status": "error",
                    "message": "All sub-claim retrievals failed.",
                }

            t0 = time.perf_counter()
            aggregated = aggregate(subclaim_results)
            timings_ms["aggregate"] = round((time.perf_counter() - t0) * 1000, 2)

            return {
                "status": "ok",
                "probability_fake": aggregated["probability_fake"],
                "verdict": aggregated["verdict"],
                "confidence_interval": aggregated["confidence_interval"],
                "evidence_trace": aggregated["evidence_trace"],
                "language": language,
                "timings_ms": timings_ms,
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Pipeline verification failed: {str(e)}",
            }

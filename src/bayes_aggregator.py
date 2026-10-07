"""Bayesian confidence score aggregator for multi-evidence claim verification."""

import math
from typing import Dict, List, Tuple


def compute_wilson_interval(p: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute 95% Wilson score confidence interval."""
    if n <= 0:
        return (0.0, 1.0)

    z = 1.95996
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def aggregate(subclaim_results: List[Dict], prior_fake: float = 0.5) -> Dict:
    """Aggregate atomic sub-claim stance predictions using iterative Bayes update.

    Args:
        subclaim_results: List of dicts with keys 'sub_claim', 'stance', and 'sources'.
        prior_fake: Prior probability that the overall claim is fake.

    Returns:
        Dict containing probability_fake, verdict, confidence_interval, and evidence_trace.
    """
    if not subclaim_results:
        p_fake = round(prior_fake, 4)
        return {
            "probability_fake": p_fake,
            "verdict": "Inconclusive",
            "confidence_interval": (0.0, 1.0),
            "evidence_trace": [],
        }

    clamped_prior = max(1e-5, min(1.0 - 1e-5, prior_fake))
    odds_f = clamped_prior / (1.0 - clamped_prior)

    evidence_trace = []
    for item in subclaim_results:
        sub_claim = item.get("sub_claim", "")
        stance = item.get("stance", {})
        sources = item.get("sources", [])

        p_support = stance.get("supports", 0.5) if isinstance(stance, dict) else 0.5
        lr = (1.0 - p_support + 1e-5) / (p_support + 1e-5)
        odds_f *= lr

        evidence_trace.append({
            "sub_claim": sub_claim,
            "stance": stance,
            "sources": sources,
        })

    p_fake = odds_f / (1.0 + odds_f)
    p_fake = round(p_fake, 4)

    if p_fake > 0.6:
        verdict = "Likely Misinformation"
    elif p_fake < 0.4:
        verdict = "Likely Real"
    else:
        verdict = "Inconclusive"

    n_subclaims = len(subclaim_results)
    ci = compute_wilson_interval(p_fake, n_subclaims)

    return {
        "probability_fake": p_fake,
        "verdict": verdict,
        "confidence_interval": ci,
        "evidence_trace": evidence_trace,
    }

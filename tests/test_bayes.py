"""Unit tests for Bayesian aggregation module."""

import pytest
from src.bayes_aggregator import aggregate


def test_bayes_all_refutes():
    subclaim_results = [
        {"sub_claim": "c1", "stance": {"supports": 0.1, "refutes": 0.9}, "sources": [{"url": "http://a.com"}]},
        {"sub_claim": "c2", "stance": {"supports": 0.1, "refutes": 0.9}, "sources": [{"url": "http://b.com"}]},
        {"sub_claim": "c3", "stance": {"supports": 0.1, "refutes": 0.9}, "sources": [{"url": "http://c.com"}]},
    ]
    res = aggregate(subclaim_results, prior_fake=0.5)
    assert res["probability_fake"] > 0.8
    assert res["verdict"] == "Likely Misinformation"
    assert len(res["evidence_trace"]) == 3


def test_bayes_all_supports():
    subclaim_results = [
        {"sub_claim": "c1", "stance": {"supports": 0.9, "refutes": 0.1}, "sources": [{"url": "http://a.com"}]},
        {"sub_claim": "c2", "stance": {"supports": 0.9, "refutes": 0.1}, "sources": [{"url": "http://b.com"}]},
        {"sub_claim": "c3", "stance": {"supports": 0.9, "refutes": 0.1}, "sources": [{"url": "http://c.com"}]},
    ]
    res = aggregate(subclaim_results, prior_fake=0.5)
    assert res["probability_fake"] < 0.2
    assert res["verdict"] == "Likely Real"
    assert len(res["evidence_trace"]) == 3


def test_bayes_wilson_interval_contains_probability():
    subclaim_results = [
        {"sub_claim": "c1", "stance": {"supports": 0.3, "refutes": 0.7}, "sources": []},
        {"sub_claim": "c2", "stance": {"supports": 0.4, "refutes": 0.6}, "sources": []},
    ]
    res = aggregate(subclaim_results, prior_fake=0.5)
    ci = res["confidence_interval"]
    p_fake = res["probability_fake"]
    assert isinstance(ci, tuple)
    assert len(ci) == 2
    assert ci[0] <= p_fake <= ci[1]

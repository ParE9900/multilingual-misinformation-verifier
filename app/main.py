"""Streamlit Dashboard for Multilingual Misinformation Verifier."""

import concurrent.futures
from pathlib import Path
import sys
import streamlit as st

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import get_config
from src.pipeline import VerificationPipeline

st.set_page_config(
    page_title="Multilingual Misinformation Verifier",
    page_icon="🔍",
    layout="centered",
)


@st.cache_resource
def get_pipeline():
    config = get_config()
    return VerificationPipeline(config=config)


def render_verdict_banner(verdict: str, p_fake: float):
    if verdict == "Likely Misinformation":
        bg_color = "#ffebee"
        border_color = "#ef5350"
        text_color = "#c62828"
        icon = "⚠️"
    elif verdict == "Likely Real":
        bg_color = "#e8f5e9"
        border_color = "#66bb6a"
        text_color = "#2e7d32"
        icon = "✅"
    else:
        bg_color = "#f5f5f5"
        border_color = "#bdbdbd"
        text_color = "#424242"
        icon = "⚖️"

    html = f"""
    <div style="background-color: {bg_color}; border: 2px solid {border_color};
                padding: 18px; border-radius: 8px; margin-bottom: 24px; text-align: center;">
        <h2 style="color: {text_color}; margin: 0; font-size: 26px;">{icon} {verdict}</h2>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_result(result: dict):
    if result.get("status") == "error":
        st.error(result.get("message", "An unknown error occurred during verification."))
        return

    verdict = result.get("verdict", "Inconclusive")
    p_fake = result.get("probability_fake", 0.5)
    ci = result.get("confidence_interval", (0.0, 1.0))
    lang = result.get("language", "unknown").upper()
    evidence_trace = result.get("evidence_trace", [])
    timings_ms = result.get("timings_ms", {})

    render_verdict_banner(verdict, p_fake)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Probability of Misinformation",
            value=f"{p_fake * 100:.1f}%",
        )
    with col2:
        st.metric(
            label="95% Confidence Interval",
            value=f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]",
        )
    with col3:
        st.metric(
            label="Detected Language",
            value=lang,
        )

    st.write("")

    with st.expander("Evidence Trace", expanded=True):
        if not evidence_trace:
            st.info("No sub-claim evidence trace available.")
        else:
            for i, item in enumerate(evidence_trace, start=1):
                sub_claim = item.get("sub_claim", "")
                stance = item.get("stance", {})
                sources = item.get("sources", [])
                p_sup = stance.get("supports", 0.5) if isinstance(stance, dict) else 0.5
                p_ref = stance.get("refutes", 0.5) if isinstance(stance, dict) else 0.5

                st.markdown(f"**{i}. Sub-Claim:** {sub_claim}")
                st.caption(f"Stance: Supports {p_sup * 100:.1f}% | Refutes {p_ref * 100:.1f}%")

                if sources and isinstance(sources, list):
                    st.markdown("**Sources:**")
                    for s in sources:
                        url = s.get("url", "")
                        title = s.get("title", "") or url or "Source"
                        snippet = s.get("snippet", "")
                        if url:
                            st.markdown(f"- [{title}]({url})")
                        if snippet:
                            st.caption(f"> \"{snippet}\"")
                else:
                    st.caption("No web sources retrieved.")
                st.divider()

    with st.expander("Pipeline Timings"):
        st.json(timings_ms)

    st.markdown("---")
    st.caption("Detector output is a probability, not a definitive verdict. Always cross-check primary sources.")


def main():
    st.title("Multilingual Misinformation Verifier")
    st.write(
        "Verify claims across multiple languages using cross-lingual NLI, "
        "Groq claim decomposition, and Gemini grounded web retrieval."
    )

    try:
        pipeline = get_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize verification pipeline: {str(e)}")
        st.stop()

    claim = st.text_area(
        label="Input Claim",
        placeholder="Enter a factual statement or claim in any supported language...",
        height=120,
    )

    if st.button("Verify Claim", type="primary"):
        clean_claim = claim.strip()
        if not clean_claim:
            st.warning("Please enter a claim to verify.")
        else:
            with st.spinner("Verifying claim... Decomposing, retrieving evidence, and computing confidence. This may take up to 60 seconds."):
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(pipeline.verify, clean_claim)
                try:
                    result = future.result(timeout=60.0)
                    st.session_state["last_result"] = result
                except concurrent.futures.TimeoutError:
                    st.session_state["last_result"] = {
                        "status": "error",
                        "message": "Verification timed out after 60 seconds.",
                    }
                except Exception as e:
                    st.session_state["last_result"] = {
                        "status": "error",
                        "message": f"Verification failed: {str(e)}",
                    }
                finally:
                    executor.shutdown(wait=False)

    if "last_result" in st.session_state:
        render_result(st.session_state["last_result"])


if __name__ == "__main__":
    main()

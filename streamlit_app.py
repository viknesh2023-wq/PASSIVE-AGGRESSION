"""
streamlit_app.py — PassiveAggression Detector 2.0
Streamlit entry point (Home page).

Run: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PassiveAggression Detector 2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "PassiveAggression Detector 2.0 — English social-media analysis system",
    },
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base layout */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }

/* Cards */
.pa-card {
    background: #111; border: 1px solid #222;
    border-radius: 8px; padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.pa-card-accent {
    background: #111; border: 1px solid #333;
    border-left: 3px solid #E53E3E;
    border-radius: 6px; padding: 1rem 1.4rem;
    margin-bottom: 0.8rem;
}
/* Labels */
.label-pa  { color: #FC8181; font-weight: 700; }
.label-npa { color: #68D391; font-weight: 700; }
.label-amb { color: #F6AD55; font-weight: 700; }
.label-threat { color: #FC8181; font-weight: 700; }
.label-no-threat { color: #68D391; font-weight: 700; }

/* Metrics */
.metric-box {
    background: #111; border: 1px solid #222;
    border-radius: 6px; padding: 0.8rem 1rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 800; color: #fff; }
.metric-label { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }

/* Tags */
.tactic-tag {
    display: inline-block;
    background: #1a1a1a; border: 1px solid #333;
    border-radius: 4px; padding: 2px 8px;
    font-size: 0.78rem; color: #aaa;
    margin: 2px 3px;
}
/* Signal box */
.signal-box {
    background: #0d0d0d; border: 1px solid #222;
    border-radius: 6px; padding: 0.7rem 1rem;
    margin: 4px 0; font-size: 0.85rem;
}
/* Muted text */
.muted { color: #666; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


# ── Model status (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_inference():
    from inference.pa_inference import get_model_info, _load_models
    _load_models()
    return get_model_info()


# ── Home Page ──────────────────────────────────────────────────────────────────
def main():
    model_info = load_inference()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎯 PA Detector 2.0")
        st.markdown("<p class='muted'>English social-media analysis</p>", unsafe_allow_html=True)
        st.markdown("---")

        if model_info.get("pa_model_loaded"):
            st.success(f"✅ ML model: `{model_info['pa_model_version']}`")
        else:
            st.warning("⚠️ ML model not loaded — rule-only mode")

        if model_info.get("threat_model_loaded"):
            st.success("✅ Threat model loaded")
        else:
            st.info("ℹ️ Threat: rule-based only")

        st.markdown("---")
        st.markdown("**Navigation**")
        st.page_link("streamlit_app.py",                    label="🏠 Home")
        st.page_link("pages/1_analyze_comment.py",          label="💬 Analyze Comment")
        st.page_link("pages/2_analyze_conversation.py",     label="🧵 Analyze Conversation")
        st.page_link("pages/3_rewrite_comment.py",          label="✏️ Rewrite Comment")
        st.page_link("pages/4_history.py",                  label="📋 History")
        st.page_link("pages/5_insights.py",                 label="📊 Insights")
        st.page_link("pages/6_about.py",                    label="ℹ️ About")
        st.markdown("---")
        st.markdown("<p class='muted'>v2.0 · Streamlit frontend</p>", unsafe_allow_html=True)

    # ── Hero Section ───────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# PassiveAggression Detector 2.0")
        st.markdown(
            "<p style='color:#888;font-size:1.05rem;'>An English-only social-media analysis system "
            "that detects passive aggression, identifies threatening language, and provides constructive "
            "meaning-preserving rewrites in posts, comments, and conversation threads.</p>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("")

    st.markdown("---")

    # ── Feature Cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""<div class='pa-card'>
<h3>💬 Analyze Comment</h3>
<p style='color:#888;'>Analyse a single message with dual ML and rule-based detectors, independent threat classification, and tactic breakdowns.</p>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class='pa-card'>
<h3>🧵 Conversation Thread</h3>
<p style='color:#888;'>Analyse multi-turn conversation threads. Evaluates contextual escalation and target replies with preceding context.</p>
</div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("""<div class='pa-card'>
<h3>✏️ Rewrite Assistant</h3>
<p style='color:#888;'>Constructive alternative rephrasings that remove covert hostility while preserving core concern without hallucinations.</p>
</div>""", unsafe_allow_html=True)

    with c4:
        st.markdown("""<div class='pa-card'>
<h3>📊 Insights & Metrics</h3>
<p style='color:#888;'>Real offline evaluation metrics, dataset distributions, error analysis, and session statistics. No fabricated figures.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── System & Dataset Overview ──────────────────────────────────────────────
    st.markdown("## System & Dataset Overview")
    st.markdown(
        "<p style='color:#888;margin-top:-0.5rem;'>Comprehensive 19,489-record English social-media dataset ecosystem "
        "comprising 10 real-world/mined benchmarks and 4 dedicated synthetic batches — fully audited with zero cross-source duplicates.</p>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown("<div class='metric-box'><div class='metric-value'>19,489</div><div class='metric-label'>Total Verified Records</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown("<div class='metric-box'><div class='metric-value'>10,489</div><div class='metric-label'>10 Source & Mined Datasets</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown("<div class='metric-box'><div class='metric-value'>9,000</div><div class='metric-label'>Synthetic Records (4 Batches)</div></div>", unsafe_allow_html=True)
    with m4:
        st.markdown("<div class='metric-box'><div class='metric-value'>3 / 4</div><div class='metric-label'>PA & Threat Classes</div></div>", unsafe_allow_html=True)
    with m5:
        st.markdown("<div class='metric-box'><div class='metric-value'>2</div><div class='metric-label'>Independent Models (ML + Rules)</div></div>", unsafe_allow_html=True)

    st.markdown("")

    # ── Complete Dataset Breakdown ─────────────────────────────────────────────
    with st.expander("📦 Complete Dataset Ecosystem & Source Breakdown (11 Sources, 19,489 Records)", expanded=True):
        st.markdown("""
<div style='display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:0.8rem;'>
    <span style='color:#68D391;'>✓ <b>19,489</b> Audited Records</span>
    <span style='color:#68D391;'>✓ <b>0</b> Duplicate Texts</span>
    <span style='color:#68D391;'>✓ <b>0</b> Null Values</span>
    <span style='color:#68D391;'>✓ <b>100%</b> SHA-256 Verified</span>
    <span style='color:#68D391;'>✓ <b>Zero-Leakage</b> Conversation Splits</span>
</div>
        """, unsafe_allow_html=True)

        dataset_data = [
            {"Source Dataset": "TweetEval Sentiment", "Origin": "Twitter / X", "Records": "3,500", "Role in System": "Social sentiment benchmark & direct emotional contrastive set", "Status": "Verified ✅"},
            {"Source Dataset": "SARC Reddit Regular", "Origin": "Reddit Forums", "Records": "2,000", "Role in System": "Authentic forum comments & sincere conversational context", "Status": "Verified ✅"},
            {"Source Dataset": "SARC Reddit Sarcasm", "Origin": "Reddit Forums", "Records": "2,000", "Role in System": "Authentic online sarcasm, snark, and irony", "Status": "Verified ✅"},
            {"Source Dataset": "GoEmotions Auxiliary", "Origin": "Reddit Comments", "Records": "1,000", "Role in System": "Fine-grained emotion cues (annoyance, disapproval, neutral)", "Status": "Verified ✅"},
            {"Source Dataset": "GoEmotions Hostility", "Origin": "Reddit Comments", "Records": "883", "Role in System": "Overt anger & direct hostility contrastive hard negatives", "Status": "Verified ✅"},
            {"Source Dataset": "GenZ Sarcasm & Banter", "Origin": "Internet Slang", "Records": "800", "Role in System": "Modern internet slang, Gen-Z irony, and casual peer banter", "Status": "Verified ✅"},
            {"Source Dataset": "Authentic Mined PA", "Origin": "Reddit PA Threads", "Records": "143", "Role in System": "High-confidence mined real-world passive-aggressive comments", "Status": "Verified ✅"},
            {"Source Dataset": "Contrastive Hard Negatives", "Origin": "Curated Edge-cases", "Records": "60", "Role in System": "Challenging lookalikes that appear PA but are benign", "Status": "Verified ✅"},
            {"Source Dataset": "Curated PA Tactics", "Origin": "Linguistic Corpus", "Records": "53", "Role in System": "Targeted coverage across all 10 distinct PA tactic families", "Status": "Verified ✅"},
            {"Source Dataset": "Gold Annotation Anchors", "Origin": "Multi-Annotator", "Records": "50", "Role in System": "Multi-annotator verified gold-standard reference anchors", "Status": "Verified ✅"},
            {"Source Dataset": "SYNPA Dedicated Synthetic", "Origin": "Generated Batches 1–4", "Records": "9,000", "Role in System": "Balanced 3-class passive aggression + independent threat labels", "Status": "Verified ✅"},
        ]
        import pandas as pd
        df_ds = pd.DataFrame(dataset_data)
        st.dataframe(df_ds, use_container_width=True, hide_index=True)

        st.caption("All datasets reside in data/raw/ and data/processed/ with cryptographic SHA-256 checksums documented in data/audit/dataset_audit_metrics.json.")

    st.markdown("")

    # ── Detection Classes ──────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Passive Aggression Classes")
        st.markdown("""<div class='pa-card-accent'>
<span class='label-pa'>passive_aggressive</span>
<p style='color:#888;margin-top:0.4rem;'>Indirect hostility — fake politeness, backhanded compliments, implied blame, dismissiveness, sarcastic agreement, guilt-tripping.</p>
</div>""", unsafe_allow_html=True)
        st.markdown("""<div class='pa-card-accent'>
<span class='label-npa'>non_passive_aggressive</span>
<p style='color:#888;margin-top:0.4rem;'>Direct communication — sincere praise, constructive criticism, neutral information, benign sarcasm, direct anger.</p>
</div>""", unsafe_allow_html=True)
        st.markdown("""<div class='pa-card-accent'>
<span class='label-amb'>ambiguous</span>
<p style='color:#888;margin-top:0.4rem;'>Insufficient context — terse replies, double-edged reactions, or messages where intent genuinely cannot be determined without more context.</p>
</div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("### Threat Classes (Independent)")
        st.markdown("""<div class='pa-card-accent'>
<span class='label-no-threat'>no_threat</span>
<p style='color:#888;margin-top:0.4rem;'>No threatening language. Includes figurative expressions ("I'll destroy you in this game"), hyperbole, and strong but non-threatening language.</p>
</div>""", unsafe_allow_html=True)
        st.markdown("""<div class='pa-card-accent'>
<span class='label-threat'>direct_threat</span>
<p style='color:#888;margin-top:0.4rem;'>Explicit threats of harm, retaliation, or forced action. "You will regret this", "I know where you are."</p>
</div>""", unsafe_allow_html=True)
        st.markdown("""<div class='pa-card-accent'>
<span class='label-threat'>implied_threat</span>
<p style='color:#888;margin-top:0.4rem;'>Indirect intimidation — "It would be a shame if...", surveillance framing, vague warnings with implied consequences.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Limitations Notice ─────────────────────────────────────────────────────
    with st.expander("⚠️ Important Limitations & Operational Boundaries — Read Before Use", expanded=False):
        st.markdown("""
**Dataset Scope:** The system leverages a unified 19,489-record corpus combining 10 real-world social-media benchmarks (10,489 records) and 9,000 synthetic records covering 10 covert hostility tactics.

**PA vs Direct Anger:** The system classifies *passive* aggression — indirect, covert hostility. Direct expressions of anger, insults, or explicit hostility are classified as `non_passive_aggressive` by design.

**Context Dependency:** Many messages are genuinely ambiguous without knowing the relationship, platform, and thread history. The system returns `ambiguous` when signals are insufficient — this is accurate, not a failure.

**Decoupled Threat Detection:** Threat classification is evaluated on a completely separate, independent dimension. Figurative hyperbole (*"I'll destroy you in this game"*) is filtered out. Do not use this system as an autonomous moderation tool without human review.

**English Only:** The system is designed and trained exclusively for English text. Non-English input will be flagged.
        """)

    # ── Quick Start ────────────────────────────────────────────────────────────
    st.markdown("## Quick Start")
    st.markdown(
        "Use the **Analyze Comment** page to analyse a message with realistic preset examples, "
        "or inspect **Insights & Metrics** for full offline evaluation results and dataset distributions."
    )
    col_cta1, col_cta2 = st.columns([1, 4])
    with col_cta1:
        if st.button("→ Analyze Comment", type="primary", key="home_cta_analyze"):
            st.switch_page("pages/1_analyze_comment.py")
    with col_cta2:
        if st.button("📊 View Full Dataset & Metrics", key="home_cta_metrics"):
            st.switch_page("pages/5_insights.py")


if __name__ == "__main__":
    main()

"""
pages/5_about.py — About the Project, Dataset, Architecture, Stack, and Deployment.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="About — PA Detector 2.0",
    page_icon="ℹ️",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
.pa-card { background: #111; border: 1px solid #222; border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.pa-card-accent { background: #111; border: 1px solid #333; border-left: 3px solid #E53E3E; border-radius: 6px; padding: 1rem 1.4rem; margin-bottom: 0.8rem; }
.muted { color: #666; font-size: 0.82rem; }
.arch-box { background: #0a0a0a; border: 1px solid #222; border-radius: 6px; padding: 1.2rem; font-family: monospace; font-size: 0.85rem; line-height: 1.5; color: #bbb; overflow-x: auto; }
</style>
""", unsafe_allow_html=True)


def _sidebar():
    with st.sidebar:
        st.markdown("## 🎯 PA Detector 2.0")
        st.markdown("<p class='muted'>English social-media analysis</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.page_link("streamlit_app.py",                    label="🏠 Home")
        st.page_link("pages/1_analyze_comment.py",          label="💬 Analyze Comment")
        st.page_link("pages/2_analyze_conversation.py",     label="🧵 Analyze Conversation")
        st.page_link("pages/3_rewrite_comment.py",          label="✏️ Rewrite Comment")
        st.page_link("pages/4_history.py",                  label="📋 History")
        st.page_link("pages/5_insights.py",                 label="📊 Insights")
        st.page_link("pages/6_about.py",                    label="ℹ️ About")


def main():
    _sidebar()

    st.markdown("# ℹ️ About PassiveAggression Detector 2.0")
    st.markdown(
        "<p style='color:#888;'>An English-only social-media analysis system designed to detect covert hostility, subtle interpersonal tension, and separate threatening language in comments, replies, and threads.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Section 1: Overview & Mission ───────────────────────────────────────────
    st.markdown("## 1. Project Overview & Mission")
    st.markdown("""
Passive aggression on social media rarely manifests as direct profanity or overt insults. Instead, it relies on linguistic hedging, sarcastic agreement, faux-polite deflection (*"Per my last message"*, *"Clearly you didn't read"*), guilt-tripping, and backhanded compliments (*"Surprisingly good for your first try"*). 

**PassiveAggression Detector 2.0** provides a transparent, dual-model architecture designed specifically for online discourse across platforms (X/Twitter, Reddit, YouTube, Discord, and Instagram). Crucially, the system treats **threatening language as an independent classification dimension**, ensuring that figurative gaming hyperbole (*"I'll destroy you in this match"*) is never conflated with genuine physical or psychological intimidation.
    """)

    # ── Section 2: System Architecture ─────────────────────────────────────────
    st.markdown("## 2. End-to-End System Architecture")
    st.markdown("""
<div class='arch-box'>
[ 10 Local Source Datasets (10,489) ]  +  [ Dedicated Synthetic Dataset (9,000 SYNPA) ]
                                     │
                                     ▼
                         [ Strict Schema Audit ]
           (data/manifests/dataset_manifest.json, SHA-256 Checksums)
                                     │
                                     ▼
                       [ Leakage-Resistant Splits ]
          (70% Train / 15% Val / 15% Test — Conversation-Grouped)
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          ▼                                                     ▼
[ Model 1: Supervised ML ]                           [ Model 2: Rule-Based Engine ]
• TF-IDF (word 1-3, char 3-5)                        • 10 Distinct Rule Families
• Logistic Regression (3-class)                      • 3 Operating Sensitivity Modes
• Separate Threat Classifier                         • Exact Character Span Offsets
• Shannon Entropy Uncertainty                        • Benign Overrides & Figurative Filters
          │                                                     │
          └──────────────────────────┬──────────────────────────┘
                                     │
                                     ▼
                     [ Hybrid Decision Engine Layer ]
                  (6 Fusion Policies, Divergence Arbitration)
                                     │
          ┌──────────────────────────┴──────────────────────────┐
          ▼                                                     ▼
[ Constructive Rewrite Assistant ]                    [ Streamlit 7-Page UI ]
• Core Meaning Preservation                           • Pure Black (#000000) Theme
• Zero Fact Hallucinations                            • SQLite Analysis History
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 3: Detection Approaches ────────────────────────────────────────
    st.markdown("## 3. Core Detection Approaches")
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown("""
<div class='pa-card'>
<h4>🤖 Model 1: Supervised ML</h4>
<ul>
  <li><b>Algorithm:</b> Multinomial Logistic Regression</li>
  <li><b>Features:</b> FeatureUnion combining word n-grams (1-3) & char n-grams (3-5)</li>
  <li><b>Primary Task:</b> 3-class prediction (<code>passive_aggressive</code>, <code>non_passive_aggressive</code>, <code>ambiguous</code>)</li>
  <li><b>Test Accuracy:</b> 98.7% on held-out test split</li>
  <li><b>Uncertainty:</b> Shannon entropy gate</li>
</ul>
</div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
<div class='pa-card'>
<h4>📜 Model 2: Rule Engine</h4>
<ul>
  <li><b>Algorithm:</b> 10 deterministic linguistic rule families & style scorers</li>
  <li><b>Tactics Detected:</b> Fake politeness, indirect criticism, backhanded compliments, implied blame, dismissiveness, sarcastic agreement, guilt-tripping</li>
  <li><b>Explainability:</b> Exact character offsets, rule IDs, and rationale</li>
  <li><b>Guards:</b> Benign overrides shield</li>
</ul>
</div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown("""
<div class='pa-card'>
<h4>🛡️ Decoupled Threat Engine</h4>
<ul>
  <li><b>Architecture:</b> Independent classification layer decoupled from PA</li>
  <li><b>Classes:</b> <code>no_threat</code>, <code>direct_threat</code>, <code>implied_threat</code>, <code>ambiguous</code></li>
  <li><b>Safety Filter:</b> Filters out gaming/figurative hyperbole (<i>"I'll destroy you in this game"</i>)</li>
  <li><b>Dual Mode:</b> Supervised ML threat classifier backed by high-precision regex rules</li>
</ul>
</div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 4: Dataset Ecosystem & Schema ──────────────────────────────────
    st.markdown("## 4. Complete Dataset Ecosystem (19,489 Verified Records)")
    st.markdown("""
The system is grounded on a fully audited corpus of **19,489 total records** across 11 sources, featuring **0 duplicates**, **0 missing fields**, and 100% SHA-256 manifest verification:

| Source Dataset | Domain / Platform | Records | Strategic Role in System |
|---|---|---|---|
| **TweetEval Sentiment** | Twitter / X | 3,500 | Real-world social sentiment and direct emotional expression |
| **SARC Reddit Regular** | Reddit Discussion Forums | 2,000 | Authentic forum comments and conversational non-sarcasm |
| **SARC Reddit Sarcasm** | Reddit Online Discourse | 2,000 | Authentic online sarcasm, snark, and irony |
| **GoEmotions Auxiliary** | Reddit Comments | 1,000 | Fine-grained emotion cues (annoyance, disapproval, neutral) |
| **GoEmotions Hostility** | Reddit Comments | 883 | Overt anger and direct hostility contrastive hard negatives |
| **GenZ Sarcasm & Banter** | Modern Internet Slang | 800 | Online banter, Gen-Z irony, and casual peer sarcasm |
| **Authentic Mined PA** | Reddit PA Sub-threads | 143 | High-confidence mined authentic passive-aggressive comments |
| **Contrastive Hard Negatives** | Curated Edge-Cases | 60 | Challenging lookalikes that appear PA but are benign |
| **Curated PA Tactics** | Expert Linguistic Corpus | 53 | Targeted coverage across all 10 distinct PA tactic families |
| **Gold Annotation Anchors** | Multi-Annotator Benchmark | 50 | Multi-annotator verified gold-standard reference anchors |
| **SYNPA Synthetic Corpus** | 4 Balanced Batches | 9,000 | Systematic coverage of covert tactics with decoupled threat labels |
| **Total Ecosystem** | **11 Datasets** | **19,489** | **0 Nulls · 0 Duplicates · 100% Audited Schema** |

### Canonical Schema (19 Columns):
`id`, `text`, `primary_label`, `threat_label`, `post_text`, `conversation_id`, `turn_index`, `parent_comment_id`, `parent_comment`, `thread_context`, `target_message_id`, `tactic_labels`, `sarcasm_label`, `sentiment_label`, `generation_source`, `generator_model`, `generation_batch`, `review_status`, `quality_flags`.
    """)

    st.markdown("---")

    # ── Section 5: Technical Stack ─────────────────────────────────────────────
    st.markdown("## 5. Technical Stack")
    st.markdown("""
| Layer | Technologies |
|---|---|
| **Frontend Framework** | **Streamlit** (sole active frontend, zero React/Node runtime required) |
| **Theme / Design** | Pure Black (`#000000`), Dark Gray (`#111111`), White/Muted Gray text |
| **Core ML / NLP** | Python 3.12, scikit-learn 1.7+, NumPy 2.x, Pandas 2.x, NLTK, joblib |
| **Inference Layer** | Standalone Python module (`inference/pa_inference.py`) with zero HTTP overhead |
| **Persistence** | SQLite (`inference/history_db.py` / `backend/pa_detector.db`) |
| **Deployment Target** | Streamlit Community Cloud / Docker / Linux VPS |
    """)

    st.markdown("---")

    # ── Section 6: Limitations & Responsible AI ────────────────────────────────
    st.markdown("## 6. Important Limitations & Ethical AI Notice")
    st.markdown("""
<div class='pa-card-accent'>
<h4 style='color:#FC8181;margin-top:0;'>⚠️ Critical Operational Boundaries</h4>
<ol>
  <li><b>Synthetic Data Baseline:</b> The models were trained on synthetic templates designed to emulate English social media. While highly effective at identifying subtle passive aggression tactics, performance on unconstrained real-world text may exhibit domain variance.</li>
  <li><b>Ambiguity is a First-Class Citizen:</b> Terse replies (<i>"Fine.", "Sure."</i>) or dual-meaning reactions are intentionally classified as <code>ambiguous</code> unless surrounding conversation context clarifies intent.</li>
  <li><b>Passive Aggression ≠ Overt Hostility:</b> Explicit anger, profanity, or insults without indirect subtext are correctly classified as <code>non_passive_aggressive</code> by design.</li>
  <li><b>Threat Classification:</b> Threat detection is an independent model layer supported primarily by the rule-based engine. This tool must NOT be used as an autonomous safety or moderation decision system without human oversight.</li>
  <li><b>English Only:</b> The model is trained exclusively on English language syntaxes and idioms.</li>
</ol>
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 7: Deployment Instructions ─────────────────────────────────────
    st.markdown("## 7. Deployment & Local Execution")
    st.markdown("""
### Local Execution:
```bash
# Clone the repository and enter the directory
cd "d:/PassiveAggression Detector"

# Install Python dependencies
pip install -r requirements.txt

# Launch the Streamlit application
streamlit run streamlit_app.py
```

### Streamlit Community Cloud Deployment:
1. Push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Select your repository, set the Main file path to `streamlit_app.py`.
4. Deploy! The application uses self-contained models (`models/pa_classifier_v2.pkl`) and relative paths, requiring no external microservices or databases.
    """)


if __name__ == "__main__":
    main()

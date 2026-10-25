"""
pages/0_unified_router.py — Unified Passive-Aggression Detection Platform.
Gemini Query Router + Social-Media Detector + Workplace Detector API.

Single statement analysis workflow that:
  1. Identifies the communication context (Social Media vs Workplace vs Uncertain) via Gemini.
  2. Dispatches to the appropriate specialized detector.
  3. Displays genuine model predictions without redirecting users or fabricating labels.
"""

import sys
import os
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from inference.unified_orchestrator import (
    analyze_unified_statement,
    check_workplace_api_health,
    get_workplace_api_url,
)
from inference.query_router import _get_gemini_api_key
from inference.pa_inference import get_model_info

st.set_page_config(
    page_title="Unified Router — PA Detector 2.0",
    page_icon="🧭",
    layout="wide",
)

# Custom Black & Charcoal Theme CSS
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
.pa-card { background: #111; border: 1px solid #222; border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.pa-card-accent { background: #111; border: 1px solid #333; border-left: 3px solid #E53E3E; border-radius: 6px; padding: 1rem 1.4rem; margin-bottom: 0.8rem; }
.router-card { background: #0c1017; border: 1px solid #1e3a5f; border-left: 4px solid #3182ce; border-radius: 6px; padding: 1.1rem 1.4rem; margin-bottom: 1.2rem; }
.workplace-card { background: #120e18; border: 1px solid #44337a; border-left: 4px solid #805ad5; border-radius: 6px; padding: 1.1rem 1.4rem; margin-bottom: 1rem; }
.badge-social { background: #1A365D; color: #63B3ED; border: 1px solid #3182ce; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.badge-workplace { background: #322659; color: #B794F4; border: 1px solid #805ad5; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.badge-uncertain { background: #443311; color: #ECC94B; border: 1px solid #D69E2E; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.badge-pa  { background: #3B1212; color: #FC8181; border: 1px solid #E53E3E; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.badge-npa { background: #0D2D17; color: #68D391; border: 1px solid #38A169; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.tactic-chip { display: inline-block; background: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; color: #aaa; margin: 2px 3px; }
.cue-chip { display: inline-block; background: #231b2e; border: 1px solid #553c9a; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; color: #d6bcfa; margin: 2px 3px; }
.muted { color: #666; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


def _render_sidebar(wp_health: dict):
    with st.sidebar:
        st.markdown("## 🎯 PA Detector 2.0")
        st.markdown("<p class='muted'>Unified Multi-Domain Platform</p>", unsafe_allow_html=True)
        st.markdown("---")

        with st.expander("⚙️ API Configuration", expanded=False):
            ui_gemini_key = st.text_input(
                "Gemini API Key (Optional)",
                type="password",
                value=st.session_state.get("cfg_gemini_key", ""),
                key="cfg_gemini_key",
                help="Paste a free Google AI Studio Gemini API key here to activate Gemini 2.0 Flash context classification.",
            )
            if ui_gemini_key:
                st.caption("🔒 Key loaded into current session.")
            else:
                st.caption("💡 No key? The built-in heuristic router classifies domains automatically!")

            wp_url = st.text_input("Workplace API URL", value=get_workplace_api_url(), key="cfg_wp_url")
            st.caption(f"Current endpoint: `{wp_url}`")

        # Router Status
        has_gemini = bool(_get_gemini_api_key())
        if has_gemini:
            st.success("✅ Gemini Router: Active (LLM)")
        else:
            st.info("ℹ️ Router: Smart Heuristic Fallback (Active)")

        # Workplace API Status
        if wp_health.get("is_available"):
            if wp_health.get("status") == "local_fallback_ready":
                st.success("✅ Workplace: Local Model Ready")
            else:
                st.success("✅ Workplace API: Ready")
        else:
            st.warning("⚠️ Workplace API: Offline")

        st.markdown("---")
        st.markdown("**Navigation**")
        st.page_link("streamlit_app.py",                    label="🏠 Home")
        st.page_link("pages/0_unified_router.py",           label="🧭 Unified Router & Detector")
        st.page_link("pages/1_analyze_comment.py",          label="💬 Analyze Comment (Social)")
        st.page_link("pages/2_analyze_conversation.py",     label="🧵 Analyze Conversation")
        st.page_link("pages/3_rewrite_comment.py",          label="✏️ Rewrite Comment")
        st.page_link("pages/4_history.py",                  label="📋 History")
        st.page_link("pages/5_insights.py",                 label="📊 Insights")
        st.page_link("pages/6_about.py",                    label="ℹ️ About")

        st.markdown("---")


def main():
    wp_health = check_workplace_api_health()
    _render_sidebar(wp_health)

    st.markdown("# 🧭 Unified Passive-Aggression Platform")
    st.markdown(
        "<p style='color:#888;font-size:1.05rem;'>Submit any statement. The <b>Gemini Context Router</b> "
        "classifies whether the communication is <b>Social Media</b> or <b>Workplace</b>, and dispatches "
        "it to the dedicated specialized detector.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Quick Example Presets ──────────────────────────────────────────────────
    c_hdr, c_clr = st.columns([5, 1])
    with c_hdr:
        st.markdown("#### 💡 Example Presets (Social, Workplace & Ambiguous)")
        st.markdown("<p class='muted' style='margin-top:-0.5rem;'>Click any preset below to test automatic domain routing:</p>", unsafe_allow_html=True)
    with c_clr:
        if st.button("🗑️ Clear", key="btn_clr_unified", use_container_width=True):
            st.session_state["unified_input_text"] = ""
            st.rerun()

    if "unified_input_text" not in st.session_state:
        st.session_state["unified_input_text"] = ""

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.caption("📱 Social-Media Examples")
        if st.button("🐦 Twitter/X Ratio Snark", key="u_tw_snark", use_container_width=True):
            st.session_state["unified_input_text"] = "Tell me you got ratioed on twitter without telling me you got ratioed."
            st.rerun()
        if st.button("💬 Reddit Downvote Hostility", key="u_rd_downvote", use_container_width=True):
            st.session_state["unified_input_text"] = "Imagine having this take in 2026. Bold of you to post this publicly, enjoy the downvotes."
            st.rerun()

    with p_col2:
        st.caption("🏢 Workplace / Corporate Examples")
        if st.button("📧 Per My Previous Email", key="u_wp_email", use_container_width=True):
            st.session_state["unified_input_text"] = "Per my previous email, as I already explained twice in yesterday's standup meeting."
            st.rerun()
        if st.button("💼 Meeting & Deadline Snark", key="u_wp_deadline", use_container_width=True):
            st.session_state["unified_input_text"] = "Glad to see someone finally reviewed the slide deck 10 minutes before the client presentation."
            st.rerun()

    with p_col3:
        st.caption("❓ Ambiguous / Uncertain Context")
        if st.button("🤐 Terse: 'Whatever you say'", key="u_amb_whatever", use_container_width=True):
            st.session_state["unified_input_text"] = "Sure, do whatever you want. It's not like my opinion matters anyway."
            st.rerun()
        if st.button("🤐 Terse: 'Fine.'", key="u_amb_fine", use_container_width=True):
            st.session_state["unified_input_text"] = "Fine."
            st.rerun()

    st.markdown("")

    # ── Input Form ─────────────────────────────────────────────────────────────
    with st.form("unified_analyze_form"):
        text_input = st.text_area(
            "Statement to analyze *",
            height=95,
            placeholder="Enter any comment, tweet, work email, Slack message, or conversation snippet…",
            help="The Gemini router will determine whether this belongs to Social Media or Workplace communication.",
            key="unified_input_text",
        )

        col_opt1, col_opt2 = st.columns([1, 1])
        with col_opt1:
            manual_override = st.selectbox(
                "Domain Selection",
                ["Automatic (Gemini Context Router)", "Force Social-Media Detector", "Force Workplace Detector"],
                index=0,
                help="Keep on Automatic to let Gemini classify the context, or manually select a detector.",
            )
        with col_opt2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            save_hist = st.checkbox("Save to session history", value=True)

        submitted = st.form_submit_button("🧭 Analyze Statement", type="primary", use_container_width=True)

    # Check for clarification button actions from previous run
    if "pending_clarify_text" in st.session_state:
        st.markdown("---")
        st.info("💡 **Clarification Requested:** Select which context you intended:")
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("📱 Evaluate as Social-Media Message", key="force_social_btn", type="primary", use_container_width=True):
                cl_text = st.session_state.pop("pending_clarify_text")
                with st.spinner("Analyzing as Social Media…"):
                    res = analyze_unified_statement(cl_text, manual_domain="social_media")
                    _render_unified_result(res)
                return
        with btn_c2:
            if st.button("🏢 Evaluate as Workplace Communication", key="force_wp_btn", type="primary", use_container_width=True):
                cl_text = st.session_state.pop("pending_clarify_text")
                with st.spinner("Analyzing as Workplace…"):
                    res = analyze_unified_statement(cl_text, manual_domain="workplace")
                    _render_unified_result(res)
                return

    # ── Form Submission ────────────────────────────────────────────────────────
    if submitted:
        input_text = text_input or st.session_state.get("unified_input_text", "")
        if not input_text or len(input_text.strip()) < 3:
            st.error("Please enter at least 3 characters to analyze.")
            return

        manual_dom = None
        if "Force Social-Media" in manual_override:
            manual_dom = "social_media"
        elif "Force Workplace" in manual_override:
            manual_dom = "workplace"

        with st.spinner("Routing communication context & analyzing…"):
            res = analyze_unified_statement(
                text=input_text,
                manual_domain=manual_dom,
                workplace_api_url=st.session_state.get("cfg_wp_url"),
            )

        if "error" in res and not res.get("is_valid", True):
            st.error(f"❌ {res['error']}")
            return

        if res.get("status") == "needs_clarification":
            st.session_state["pending_clarify_text"] = input_text
            st.markdown("---")
            r_info = res.get("router_info", {})
            st.markdown(f"""
            <div class='router-card' style='border-left-color:#D69E2E;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                        <span class='badge-uncertain'>❓ Context Uncertain / Ambiguous</span>
                        <h4 style='color:#fff; margin:0.4rem 0 0.2rem 0;'>Clarification Required</h4>
                        <p style='color:#ccc; font-size:0.9rem; margin:0;'>{r_info.get('explanation')}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("Please specify which context you would like to evaluate this statement under:")
            btn_c1, btn_c2 = st.columns(2)
            with btn_c1:
                if st.button("📱 Evaluate as Social-Media Message", key="cl_soc_btn", type="primary", use_container_width=True):
                    res_m = analyze_unified_statement(input_text, manual_domain="social_media")
                    _render_unified_result(res_m)
            with btn_c2:
                if st.button("🏢 Evaluate as Workplace Communication", key="cl_wp_btn", type="primary", use_container_width=True):
                    res_m = analyze_unified_statement(input_text, manual_domain="workplace")
                    _render_unified_result(res_m)
            return

        # Render complete result
        _render_unified_result(res)


def _render_unified_result(res: dict):
    """Render the unified routing and detection result."""
    st.markdown("---")
    st.markdown("## Analysis Results")

    domain = res.get("selected_domain", "social_media")
    router_info = res.get("router_info", {})
    detector_info = res.get("detector_info", {})
    pred = res.get("prediction", {})
    lat = res.get("total_latency_ms", 0.0)

    # ── 1. Gemini Router Decision Banner ───────────────────────────────────────
    if domain == "social_media":
        badge_html = "<span class='badge-social'>📱 Social Media Domain</span>"
        border_col = "#3182ce"
    else:
        badge_html = "<span class='badge-workplace'>🏢 Workplace Domain</span>"
        border_col = "#805ad5"

    router_source = router_info.get("router_model", "gemini-2.0-flash")
    if router_info.get("is_fallback"):
        router_badge = f"<span class='muted' style='font-size:0.75rem;'>Router: {router_source} (Heuristic)</span>"
    else:
        router_badge = f"<span style='color:#63B3ED;font-size:0.75rem;font-weight:700;'>Router: {router_source} (LLM)</span>"

    st.markdown(f"""
    <div class='router-card' style='border-left-color:{border_col};'>
        <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
            <div>
                {badge_html}
                <div style='margin-top:0.5rem;'>
                    <b style='color:#fff;'>Context Assessment:</b> <span style='color:#ddd;'>{router_info.get('explanation')}</span>
                </div>
            </div>
            <div style='text-align:right;'>
                {router_badge}<br>
                <span class='muted'>Confidence: {int(router_info.get('confidence', 0)*100)}%</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. Detector Identity & Metrics ─────────────────────────────────────────
    det_name = detector_info.get("detector_name", "Specialized Model")
    arch = detector_info.get("architecture_type", "")
    st.markdown(f"### Invoked Detector: **{det_name}**")
    st.markdown(f"<p class='muted' style='margin-top:-0.5rem;'>Architecture: <code>{arch}</code> · Execution latency: {lat:.1f} ms</p>", unsafe_allow_html=True)

    # ── 3. Domain-Specific Result Cards ────────────────────────────────────────
    if domain == "social_media":
        _render_social_output(pred)
    elif domain == "workplace":
        _render_workplace_output(pred)


def _render_social_output(pred: dict):
    """Render authentic Social-Media Detector outputs (Model 1 ML + Model 2 Rules + Threat)."""
    col1, col2, col3 = st.columns(3)

    label = pred.get("primary_label", "non_passive_aggressive")
    score = pred.get("score", 0.0)

    with col1:
        st.markdown("#### Model 1: Supervised ML")
        if label == "passive_aggressive":
            st.markdown("<span class='badge-pa'>passive_aggressive</span>", unsafe_allow_html=True)
        elif label == "non_passive_aggressive":
            st.markdown("<span class='badge-npa'>non_passive_aggressive</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-amb'>ambiguous</span>", unsafe_allow_html=True)
        st.metric("ML PA Score", f"{score:.1f} / 100")
        st.caption("Derived from 3-class TF-IDF Logistic Regression")

    with col2:
        rule_lbl = pred.get("rule_label", "non_passive_aggressive")
        rule_sc = pred.get("rule_score", 0.0)
        st.markdown("#### Model 2: Rule Detector")
        if rule_lbl == "passive_aggressive":
            st.markdown("<span class='badge-pa'>passive_aggressive</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-npa'>non_passive_aggressive</span>", unsafe_allow_html=True)
        st.metric("Rule Score", f"{rule_sc:.1f} / 100")
        if pred.get("models_agree"):
            st.success("✅ Models in Agreement")
        else:
            st.info("ℹ️ Models Diverge")

    with col3:
        threat = pred.get("threat", {})
        t_lbl = threat.get("label", "no_threat")
        st.markdown("#### Decoupled Threat Engine")
        if threat.get("is_flagged"):
            st.markdown(f"<span class='badge-pa'>{t_lbl}</span>", unsafe_allow_html=True)
            st.warning("⚠️ Threat flagged independently")
        else:
            st.markdown("<span class='badge-npa'>no_threat</span>", unsafe_allow_html=True)
            st.success("✅ No threat detected")
        st.caption("Decoupled from PA classification")

    # Tactics Chips
    tactics = pred.get("tactics", [])
    if tactics:
        st.markdown("---")
        st.markdown("**Detected Social-Media Tactics:**")
        st.markdown("".join(f"<span class='tactic-chip'>{t.replace('_',' ')}</span>" for t in tactics), unsafe_allow_html=True)


def _render_workplace_output(pred: dict):
    """Render authentic Workplace Detector outputs (Binary Logistic Regression + VADER)."""
    col1, col2, col3 = st.columns(3)

    raw_label = pred.get("primary_label", "Not flagged as passive-aggressive")
    is_pa = "Potentially" in raw_label or pred.get("binary_prediction") == 1
    score = pred.get("score", 0.0)

    with col1:
        st.markdown("#### Workplace Prediction")
        if is_pa:
            st.markdown("<span class='badge-pa'>Potentially passive-aggressive</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-npa'>Not flagged as passive-aggressive</span>", unsafe_allow_html=True)
        st.metric("Model Probability", f"{score:.1f}%")
        st.caption("Estimated probability for predicted class (abijitabi47 model)")

    with col2:
        st.markdown("#### Supporting Sentiment (VADER)")
        s_lbl = pred.get("sentiment_label", "Neutral")
        s_score = pred.get("sentiment_score", 0.0)
        st.markdown(f"**{s_lbl}**")
        st.metric("Compound Score", f"{s_score:.3f}")
        st.caption("Supporting polarity signal (not intent)")

    with col3:
        st.markdown("#### Threat Evaluation")
        st.markdown("<span class='muted'>Not applicable in workplace model</span>", unsafe_allow_html=True)
        st.metric("Threat Status", "N/A")
        st.caption("Workplace model evaluates covert hostility only")

    # Model Weighted Cues
    cues = pred.get("cues", [])
    if cues:
        st.markdown("---")
        st.markdown("**Model-Weighted Workplace Cues:**")
        st.markdown("".join(f"<span class='cue-chip'>{cue}</span>" for cue in cues), unsafe_allow_html=True)
        st.caption("Top n-grams contributing to the logistic regression classification")


if __name__ == "__main__":
    main()

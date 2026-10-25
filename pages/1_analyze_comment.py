"""
pages/1_analyze_comment.py — Analyze a single social-media comment.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Analyze Comment — PA Detector 2.0",
    page_icon="💬",
    layout="wide",
)

# Shared CSS
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
.pa-card { background: #111; border: 1px solid #222; border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.result-badge-pa  { background: #3B1212; color: #FC8181; border: 1px solid #E53E3E; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.9rem; }
.result-badge-npa { background: #0D2D17; color: #68D391; border: 1px solid #38A169; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.9rem; }
.result-badge-amb { background: #2D2010; color: #F6AD55; border: 1px solid #D69E2E; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.9rem; }
.result-badge-thr { background: #3B1212; color: #FC8181; border: 1px solid #E53E3E; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.result-badge-ok  { background: #0D2D17; color: #68D391; border: 1px solid #38A169; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.signal-row { background: #0d0d0d; border: 1px solid #222; border-radius: 6px; padding: 0.6rem 1rem; margin: 3px 0; font-size: 0.85rem; }
.tactic-chip { display: inline-block; background: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 2px 8px; font-size: 0.78rem; color: #aaa; margin: 2px 3px; }
.muted { color: #666; font-size: 0.82rem; }
.disc-box { background: #0d0d0d; border: 1px solid #333; border-left: 3px solid #D69E2E; border-radius: 6px; padding: 0.7rem 1rem; font-size: 0.83rem; color: #888; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def _load():
    from inference.pa_inference import _load_models, get_model_info
    _load_models()
    return get_model_info()


def _label_badge(label: str) -> str:
    cls_map = {
        "passive_aggressive":     "result-badge-pa",
        "non_passive_aggressive": "result-badge-npa",
        "ambiguous":              "result-badge-amb",
        "direct_threat":          "result-badge-thr",
        "implied_threat":         "result-badge-thr",
        "no_threat":              "result-badge-ok",
    }
    display = label.replace("_", " ").title()
    css     = cls_map.get(label, "result-badge-amb")
    return f"<span class='{css}'>{display}</span>"


def _render_sidebar(model_info):
    with st.sidebar:
        st.markdown("## 🎯 PA Detector 2.0")
        st.markdown("<p class='muted'>English social-media analysis</p>", unsafe_allow_html=True)
        st.markdown("---")
        if model_info.get("pa_model_loaded"):
            st.success(f"✅ `{model_info['pa_model_version']}`")
        else:
            st.warning("⚠️ Rule-only mode")
        st.markdown("---")
        st.page_link("streamlit_app.py",                    label="🏠 Home")
        st.page_link("pages/0_unified_router.py",           label="🧭 Unified Router & Detector")
        st.page_link("pages/1_analyze_comment.py",          label="💬 Analyze Comment (Social)")
        st.page_link("pages/2_analyze_conversation.py",     label="🧵 Analyze Conversation")
        st.page_link("pages/3_rewrite_comment.py",          label="✏️ Rewrite Comment")
        st.page_link("pages/4_history.py",                  label="📋 History")
        st.page_link("pages/5_insights.py",                 label="📊 Insights")
        st.page_link("pages/6_about.py",                    label="ℹ️ About")


def main():
    model_info = _load()
    _render_sidebar(model_info)

    st.markdown("# 💬 Analyze Comment")
    st.markdown(
        "<p style='color:#888;'>Analyse a single social-media message for passive aggression and threatening language.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Social Media Presets ───────────────────────────────────────────────────
    c_hdr, c_clr = st.columns([5, 1])
    with c_hdr:
        st.markdown("#### 📱 Social-Media Presets (Click to test)")
        st.markdown("<p class='muted' style='margin-top:-0.5rem;'>Click any social-media preset below to auto-populate target message, original post, and parent context:</p>", unsafe_allow_html=True)
    with c_clr:
        if st.button("🗑️ Clear All", key="clear_preset_btn", use_container_width=True):
            st.session_state["comment_text_area"] = ""
            st.session_state["post_text_area"] = ""
            st.session_state["parent_text_area"] = ""
            st.rerun()

    if "comment_text_area" not in st.session_state:
        st.session_state["comment_text_area"] = ""
    if "post_text_area" not in st.session_state:
        st.session_state["post_text_area"] = ""
    if "parent_text_area" not in st.session_state:
        st.session_state["parent_text_area"] = ""

    # Social Media Scenario Buttons (Populates Target + Post + Parent in 1 click)
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        if st.button("🐦 Twitter/X: Quote Snark", key="p_twitter_snark", use_container_width=True, help="Tell me you didn't read the article without telling me..."):
            st.session_state["comment_text_area"] = "Tell me you didn't read the article without telling me you didn't read the article."
            st.session_state["post_text_area"] = "New study reveals major shifts in social media algorithms and user engagement patterns."
            st.session_state["parent_text_area"] = "This entire article makes zero sense, complete waste of time."
            st.rerun()
        if st.button("💬 Reddit: Downvote / Sarcasm", key="p_reddit_downvote", use_container_width=True, help="Imagine having this take in 2026. Bold move."):
            st.session_state["comment_text_area"] = "Imagine having this take in 2026. Bold of you to post this publicly, enjoy the downvotes."
            st.session_state["post_text_area"] = "Discussion Thread: Which gaming updates this year actually improved player retention?"
            st.session_state["parent_text_area"] = "Honestly the old legacy version was 100 times better and the devs are clueless."
            st.rerun()

    with p_col2:
        if st.button("📸 Instagram: Backhanded Praise", key="p_insta_backhanded", use_container_width=True, help="You look so confident wearing that..."):
            st.session_state["comment_text_area"] = "You look so confident wearing that! I could never be brave enough to pull off something so unflattering."
            st.session_state["post_text_area"] = "Weekend outfit drop! Stepping outside my comfort zone with this new look."
            st.session_state["parent_text_area"] = "You look absolutely stunning queen! 🔥"
            st.rerun()
        if st.button("🎮 Discord / Gaming: Resentment", key="p_gaming_resentment", use_container_width=True, help="Must be nice getting carried to Diamond rank..."):
            st.session_state["comment_text_area"] = "Must be nice getting carried to Diamond rank while the rest of us actually play the objective."
            st.session_state["post_text_area"] = "Tournament Team Announcements: Final roster locked in for the season playoffs."
            st.session_state["parent_text_area"] = "Just got promoted to Diamond support! Let's win this tournament!"
            st.rerun()

    with p_col3:
        if st.button("✨ Social Media: Sincere Praise", key="p_sincere_social", use_container_width=True, help="Genuinely love this breakdown! Super helpful..."):
            st.session_state["comment_text_area"] = "Genuinely love this breakdown! Super helpful explanation, thanks for sharing."
            st.session_state["post_text_area"] = "Quick 5-step video tutorial on optimizing stream audio and background noise reduction."
            st.session_state["parent_text_area"] = "Does this configuration also work for USB condenser microphones?"
            st.rerun()
        if st.button("🚨 Direct Threat: Safety Flag", key="p_direct_threat", use_container_width=True, help="Keep talking like that and I will find where you live..."):
            st.session_state["comment_text_area"] = "Keep talking like that and I will find where you live and make sure you pay for this."
            st.session_state["post_text_area"] = "Streaming our charity gaming marathon live all weekend from downtown."
            st.session_state["parent_text_area"] = "See you in the stream chat tomorrow!"
            st.rerun()

    st.markdown("")

    # ── Input Form ─────────────────────────────────────────────────────────────
    with st.form("analyze_form"):
        text_input = st.text_area(
            "Message to analyse *",
            height=100,
            placeholder="Paste a social-media comment, reply, or post here, or select one of the presets above…",
            help="English social-media text only. Min 3 characters.",
            key="comment_text_area",
        )

        col1, col2 = st.columns(2)
        with col1:
            post_text = st.text_area(
                "Original post (optional context)",
                height=70,
                placeholder="The post, tweet, or video this message is responding to…",
                key="post_text_area",
            )
        with col2:
            parent_comment = st.text_area(
                "Parent comment (optional context)",
                height=70,
                placeholder="The specific preceding comment or reply being answered…",
                key="parent_text_area",
            )

        col_sens, col_hist = st.columns([1, 1])
        with col_sens:
            sensitivity = st.selectbox(
                "Sensitivity",
                ["balanced", "strict", "lenient"],
                index=0,
                help="Strict = more PA detections; Lenient = fewer",
            )
        with col_hist:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            save_to_history = st.checkbox("Save to history", value=True)

        context = "social"
        submitted = st.form_submit_button("🔍 Analyse", type="primary", use_container_width=True)

    # ── Validation ─────────────────────────────────────────────────────────────
    if submitted:
        input_to_check = text_input or st.session_state.get("comment_text_area", "")
        if not input_to_check or len(input_to_check.strip()) < 3:
            st.error("Please enter at least 3 characters of text to analyse.")
            return

        if len(input_to_check) > 2000:
            st.warning(f"Text is {len(input_to_check)} chars. Truncating to 2,000 characters for analysis.")
            input_to_check = input_to_check[:2000]

        with st.spinner("Analysing…"):
            from inference.pa_inference import analyze
            result = analyze(
                text=input_to_check,
                context=context,
                sensitivity=sensitivity,
                post_text=post_text or None,
                parent_comment=parent_comment or None,
            )

        if "error" in result:
            st.error(f"❌ {result['error']}")
            return

        # ── Save to history ────────────────────────────────────────────────────
        if save_to_history:
            try:
                from inference.history_db import save as save_history
                save_history(result["analysis_id"], input_to_check, context, sensitivity, result)
            except Exception as he:
                st.warning(f"History save failed: {he}")

        _render_results(result, input_to_check, model_info)


def _render_results(result: dict, text_input: str, model_info: dict):
    """Render the analysis results."""
    st.markdown("---")
    st.markdown("## Results")

    ml     = result.get("ml_result", {})
    rule   = result.get("rule_result", {})
    threat = result.get("threat_result", {})
    comb   = result.get("combined", {})
    sents  = result.get("sentence_scores", [])

    analysis_id = result.get("analysis_id", "?")
    latency_ms  = result.get("total_latency_ms", 0)

    # ── Top summary bar ────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='color:#555;font-size:0.8rem;margin-bottom:0.8rem;'>"
        f"Analysis ID: <code>{analysis_id}</code> · Latency: {latency_ms:.1f} ms"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Main inspectable verdict columns (Model 1, Model 2, Threat) ───────────
    col_ml, col_rule, col_threat = st.columns(3)

    with col_ml:
        ml_label = ml.get("label", "non_passive_aggressive")
        ml_score = ml.get("pa_score", 0)
        ml_mode  = ml.get("inference_mode", "?")
        unc      = ml.get("uncertainty", {})

        st.markdown("#### Model 1: Supervised ML")
        st.markdown(_label_badge(ml_label), unsafe_allow_html=True)
        st.metric("ML PA Score", f"{ml_score:.1f} / 100")

        if model_info.get("pa_model_loaded"):
            probs = ml.get("probabilities", {})
            if probs:
                for lbl, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    st.markdown(
                        f"<p style='font-size:0.8rem;color:#888;margin:1px 0;'>"
                        f"{lbl.replace('_',' ')}: <b style='color:#ccc'>{prob:.3f}</b></p>",
                        unsafe_allow_html=True,
                    )

        if unc.get("is_uncertain"):
            st.warning(f"⚠️ Uncertain: {unc.get('note','')}")

        st.markdown(f"<p class='muted'>Mode: {ml_mode}</p>", unsafe_allow_html=True)

    with col_rule:
        rule_label = rule.get("pa_label", "non_passive_aggressive")
        rule_score = rule.get("pa_score", 0)
        tactics    = rule.get("tactics", [])

        st.markdown("#### Model 2: Rule Detector")
        st.markdown(_label_badge(rule_label), unsafe_allow_html=True)
        st.metric("Rule Score", f"{rule_score:.1f} / 100")

        if tactics:
            st.markdown("**Detected tactics:**")
            chips = "".join(
                f"<span class='tactic-chip'>{t.replace('_',' ')}</span>"
                for t in tactics[:5]
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown("<p class='muted'>No tactics detected</p>", unsafe_allow_html=True)

        agree = ml.get("label") == rule.get("pa_label")
        if agree:
            st.success("✅ Models in Agreement")
        else:
            st.info("ℹ️ Model Divergence: ML and Rule models differ — review both signals above")

    with col_threat:
        threat_label = threat.get("label", "no_threat")
        threat_mode  = threat.get("inference_mode", "rule_based")
        rule_t_score = rule.get("threat_score", 0)

        st.markdown("#### Threat Detection (Independent)")
        st.markdown(_label_badge(threat_label), unsafe_allow_html=True)

        if threat.get("probabilities") and threat_mode == "ml_threat":
            t_conf = threat.get("confidence", 0)
            st.metric("ML Confidence", f"{t_conf:.3f}")
        else:
            st.metric("Threat Score", f"{rule_t_score:.1f} / 100")
            st.markdown(f"<p class='muted'>Mode: rule-based</p>", unsafe_allow_html=True)

        if threat_label != "no_threat":
            st.warning(f"⚠️ Threat flagged: {threat_label.replace('_',' ')}")
        else:
            st.success("✅ No threat detected")

    # ── Rule-Based Signals ─────────────────────────────────────────────────────
    st.markdown("---")
    signals = rule.get("signals", [])
    threat_signals = rule.get("threat_signals", [])

    if signals or threat_signals:
        with st.expander(f"📋 Rule Signals ({len(signals)} PA + {len(threat_signals)} threat)", expanded=True):
            if signals:
                st.markdown("**Passive-aggression signals:**")
                for sig in signals[:8]:
                    tactic = sig.get("tactic", "").replace("_", " ")
                    st.markdown(
                        f"<div class='signal-row'>"
                        f"<b style='color:#aaa;'>\"{sig.get('phrase','')}\"</b> "
                        f"<span class='tactic-chip'>{tactic}</span> "
                        f"<span style='color:#666;'>weight={sig.get('weight',0):.0f}</span><br>"
                        f"<span style='color:#777;'>{sig.get('reason','')}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            if threat_signals:
                st.markdown("**Threat signals:**")
                for sig in threat_signals[:5]:
                    st.markdown(
                        f"<div class='signal-row' style='border-left:2px solid #E53E3E;'>"
                        f"<b style='color:#FC8181;'>\"{sig.get('phrase','')}\"</b> "
                        f"<span style='color:#666;'>weight={sig.get('weight',0):.0f}</span><br>"
                        f"<span style='color:#777;'>{sig.get('reason','')}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
    else:
        with st.expander("📋 Rule Signals"):
            st.markdown("<p class='muted'>No rule-based signals matched in this text.</p>", unsafe_allow_html=True)

    # ── Explanation ────────────────────────────────────────────────────────────
    explanation = comb.get("explanation", "")
    if explanation:
        st.markdown("---")
        st.markdown("**Explanation:**")
        st.markdown(
            f"<div class='disc-box'>{explanation}</div>",
            unsafe_allow_html=True,
        )

    # ── Flags ──────────────────────────────────────────────────────────────────
    flags = comb.get("flags", [])
    if flags:
        for flag in flags:
            if "Threat" in flag or "⚠️" in flag:
                st.warning(flag)
            else:
                st.info(flag)

    # ── Highlighted Matched Text ───────────────────────────────────────────────
    if signals:
        st.markdown("---")
        st.markdown("**Highlighted Analysis:**")
        highlighted_html = text_input
        # Sort signals by length descending to avoid partial tag collisions
        sorted_sigs = sorted(signals, key=lambda s: len(s.get("phrase", "")), reverse=True)
        import html
        highlighted_html = html.escape(highlighted_html)
        for sig in sorted_sigs:
            phrase = sig.get("phrase", "")
            if phrase and phrase.lower() in text_input.lower():
                escaped_phrase = html.escape(phrase)
                # replace case-insensitively
                import re
                highlighted_html = re.sub(
                    re.escape(escaped_phrase),
                    f"<mark style='background:#5c1d1d;color:#ffc9c9;padding:2px 5px;border-radius:3px;'>{escaped_phrase}</mark>",
                    highlighted_html,
                    flags=re.IGNORECASE,
                )
        st.markdown(
            f"<div class='pa-card' style='font-size:1.05rem;line-height:1.6;'>{highlighted_html}</div>",
            unsafe_allow_html=True,
        )

    # ── Constructive Rewrite Action ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ✏️ Constructive Alternative Rewriting")
    from inference.rewrite_engine import rewrite
    rewrite_res = rewrite(text_input, parent_comment=None, detection_result=result)

    if rewrite_res.has_rewrites and rewrite_res.candidates:
        st.markdown(
            f"<p style='color:#a0aec0;'>Detected {rewrite_res.primary_tactic.replace('_',' ') if rewrite_res.primary_tactic else 'indirect'} framing. "
            "Here is an instant constructive suggestion to communicate directly without indirect hostility:</p>",
            unsafe_allow_html=True,
        )
        cand = rewrite_res.candidates[0]
        st.markdown(
            f"<div class='signal-row' style='border-left:3px solid #38A169;background:#0d1a10;padding:0.9rem;'>"
            f"<b style='color:#68D391;'>Suggested:</b> \"<span style='color:#fff;'>{cand.suggested_text}</span>\"<br>"
            f"<span style='color:#888;font-size:0.8rem;'>Rationale: {cand.explanation}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        col_rw1, col_rw2 = st.columns([1, 3])
        with col_rw1:
            if st.button("✏️ Open in Full Rewrite Assistant"):
                st.session_state["rewrite_prefill_text"] = text_input
                st.switch_page("pages/3_rewrite_comment.py")
    elif rewrite_res.is_already_respectful:
        st.markdown("<p style='color:#68D391;font-size:0.85rem;'>✅ Message is already direct and respectful. No rewrite needed.</p>", unsafe_allow_html=True)
    elif rewrite_res.is_threat_flagged:
        st.markdown("<p style='color:#FC8181;font-size:0.85rem;'>⚠️ Threat flagged in message. Rewriting is withheld for safety.</p>", unsafe_allow_html=True)

    # ── Context reminder ───────────────────────────────────────────────────────
    st.markdown(
        "<div class='disc-box'>ℹ️ Results reflect analysis of the <b>target message only</b> plus any provided context. "
        "The system does not have access to external platform data or user history. "
        "Both ML and rule-based results are derived independently — neither automatically overrides the other.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

"""
pages/3_rewrite_comment.py — Constructive, meaning-preserving rewrite assistant.
Transforms passive-aggressive social-media messages into clear, direct, and respectful communication.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from inference.rewrite_engine import rewrite, RewriteResponse
from inference.pa_inference import analyze, get_model_info

st.set_page_config(
    page_title="Rewrite Comment — PA Detector 2.0",
    page_icon="✏️",
    layout="wide",
)

# Custom Black & Charcoal Theme CSS
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
.pa-card { background: #111; border: 1px solid #222; border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
.pa-card-accent { background: #111; border: 1px solid #333; border-left: 3px solid #E53E3E; border-radius: 6px; padding: 1rem 1.4rem; margin-bottom: 0.8rem; }
.rewrite-box { background: #0c140e; border: 1px solid #276749; border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
.orig-box { background: #140c0c; border: 1px solid #742a2a; border-radius: 6px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
.badge-pa  { background: #3B1212; color: #FC8181; border: 1px solid #E53E3E; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.badge-npa { background: #0D2D17; color: #68D391; border: 1px solid #38A169; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.badge-amb { background: #2D2010; color: #F6AD55; border: 1px solid #D69E2E; border-radius: 4px; padding: 3px 10px; font-weight: 700; font-size: 0.85rem; }
.tactic-chip { display: inline-block; background: #1a1a1a; border: 1px solid #333; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; color: #aaa; margin-right: 4px; }
.muted { color: #666; font-size: 0.82rem; }
.disclaimer-box { background: #0a0a0a; border: 1px solid #333; border-left: 3px solid #718096; border-radius: 6px; padding: 0.8rem 1rem; font-size: 0.83rem; color: #888; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


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
        st.markdown("**Navigation**")
        st.page_link("streamlit_app.py",                    label="🏠 Home")
        st.page_link("pages/0_unified_router.py",           label="🧭 Unified Router & Detector")
        st.page_link("pages/1_analyze_comment.py",          label="💬 Analyze Comment (Social)")
        st.page_link("pages/2_analyze_conversation.py",     label="🧵 Analyze Conversation")
        st.page_link("pages/3_rewrite_comment.py",          label="✏️ Rewrite Comment")
        st.page_link("pages/4_history.py",                  label="📋 History")
        st.page_link("pages/5_insights.py",                 label="📊 Insights")
        st.page_link("pages/6_about.py",                    label="ℹ️ About")


def main():
    model_info = get_model_info()
    _render_sidebar(model_info)

    st.markdown("# ✏️ Rewrite Comment")
    st.markdown(
        "<p style='color:#888;'>Transform covertly hostile, indirect, or dismissive text into clear, "
        "respectful language while strictly preserving your core communicative concern.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── State Initialization ───────────────────────────────────────────────────
    if "rewrite_comment_input" not in st.session_state:
        st.session_state["rewrite_comment_input"] = st.session_state.get("rewrite_prefill_text", "")
    if "rewrite_parent_input" not in st.session_state:
        st.session_state["rewrite_parent_input"] = st.session_state.get("rewrite_prefill_parent", "")

    # If navigated from another page with prefill
    if st.session_state.get("rewrite_prefill_text"):
        st.session_state["rewrite_comment_input"] = st.session_state.pop("rewrite_prefill_text")
    if st.session_state.get("rewrite_prefill_parent"):
        st.session_state["rewrite_parent_input"] = st.session_state.pop("rewrite_prefill_parent")

    # ── Social Media Presets for Rewriting ──────────────────────────────────────
    c_hdr, c_clr = st.columns([5, 1])
    with c_hdr:
        st.markdown("#### 📱 Social-Media Presets (Click to test)")
        st.markdown("<p class='muted' style='margin-top:-0.5rem;'>Click any preset below to auto-populate a covertly hostile social-media comment and its context:</p>", unsafe_allow_html=True)
    with c_clr:
        if st.button("🗑️ Clear", key="clear_rewrite_btn", use_container_width=True):
            st.session_state["rewrite_comment_input"] = ""
            st.session_state["rewrite_parent_input"] = ""
            st.rerun()

    r1, r2, r3 = st.columns(3)
    with r1:
        if st.button("🐦 Twitter/X: Snarky Gatekeeping", key="rw_tw_snark", use_container_width=True, help="Tell me you didn't read the article..."):
            st.session_state["rewrite_comment_input"] = "Tell me you didn't bother reading the article without actually saying it."
            st.session_state["rewrite_parent_input"] = "Why is everyone upset about the new algorithm update? Seems totally fine to me."
            st.rerun()
        if st.button("📸 Insta/YouTube: Backhanded", key="rw_insta_bh", use_container_width=True, help="Surprisingly decent video for a small channel..."):
            st.session_state["rewrite_comment_input"] = "Surprisingly decent video, honestly didn't expect much from a channel with under 10k subscribers."
            st.session_state["rewrite_parent_input"] = "Just published my first 30-minute mini-documentary! Feedback appreciated."
            st.rerun()

    with r2:
        if st.button("💬 Reddit: Grudging Acquiescence", key="rw_reddit_grudge", use_container_width=True, help="Whatever you say boss..."):
            st.session_state["rewrite_comment_input"] = "Whatever you say boss. You're always right and everyone else is wrong."
            st.session_state["rewrite_parent_input"] = "I'm locking this subthread because the debate has moved off-topic."
            st.rerun()
        if st.button("📱 TikTok: Sponsor Resentment", key="rw_tiktok_resent", use_container_width=True, help="Must be nice getting free creator packages..."):
            st.session_state["rewrite_comment_input"] = "Must be nice getting free creator packages while real fans pay full price like suckers."
            st.session_state["rewrite_parent_input"] = "Unboxing the ultimate gaming rig sent over by our title sponsor!"
            st.rerun()

    with r3:
        if st.button("🎮 Discord: Sarcastic Blame", key="rw_discord_blame", use_container_width=True, help="Thanks for throwing the final round..."):
            st.session_state["rewrite_comment_input"] = "Thanks for throwing the final round, really appreciate your stellar gameplay today."
            st.session_state["rewrite_parent_input"] = "Sorry team, completely missed that last shot."
            st.rerun()
        if st.button("💬 Sarcastic False Agreement", key="rw_sarc_agree", use_container_width=True, help="Oh yeah, because that strategy worked SO well..."):
            st.session_state["rewrite_comment_input"] = "Oh yeah, because that strategy worked SO well for the team in the last tournament."
            st.session_state["rewrite_parent_input"] = "I think we should stick to the exact same comp we used yesterday."
            st.rerun()

    st.markdown("")

    # Input Form
    with st.form("rewrite_form"):
        comment_input = st.text_area(
            "Original Comment to Rewrite *",
            height=90,
            placeholder="e.g. Tell me you didn't read the article without telling me you didn't read it…",
            help="English social-media text. The assistant preserves your core meaning.",
            key="rewrite_comment_input",
        )
        parent_input = st.text_area(
            "Preceding / Parent Context (optional)",
            height=70,
            placeholder="The post or comment being responded to…",
            help="Helps the assistant understand context for terse replies like 'fine' or 'sure'.",
            key="rewrite_parent_input",
        )

        col_btn, _ = st.columns([2, 3])
        with col_btn:
            submit = st.form_submit_button("✨ Generate Rewrites", type="primary", use_container_width=True)

    if submit:
        if not comment_input or not comment_input.strip():
            st.error("Please enter a comment to rewrite.")
            return

        with st.spinner("Analyzing tone and generating constructive alternatives…"):
            # 1. Run detection for rich context
            detection = analyze(
                text=comment_input.strip(),
                parent_comment=parent_input.strip() if parent_input.strip() else None,
                sensitivity="balanced",
            )
            # 2. Run rewrite engine
            rewrite_res: RewriteResponse = rewrite(
                text=comment_input.strip(),
                parent_comment=parent_input.strip() if parent_input.strip() else None,
                detection_result=detection,
            )

        st.markdown("---")
        st.markdown("## Rewrite Results")

        # Top verdict card
        ml_res = detection.get("ml_result", {})
        rule_res = detection.get("rule_result", {})
        tactics = rule_res.get("tactics", [])
        pa_label = detection.get("combined", {}).get("verdict", ml_res.get("label", "non_passive_aggressive"))

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### 📝 Original Message")
            st.markdown(
                f"<div class='orig-box'><p style='color:#FC8181;margin:0;'>{comment_input.strip()}</p></div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Detected Label:** `{pa_label}`")
            if tactics:
                chips = " ".join([f"<span class='tactic-chip'>{t.replace('_', ' ')}</span>" for t in tactics])
                st.markdown(f"**Tactics Identified:** {chips}", unsafe_allow_html=True)

        with col_right:
            st.markdown("### 💡 Analysis & Rationale")
            st.markdown(f"<div class='pa-card'><p style='color:#ddd;margin:0;'>{rewrite_res.rationale}</p></div>", unsafe_allow_html=True)
            if rewrite_res.is_threat_flagged:
                st.error("⚠️ Threat detected in original message. Rewriting is withheld for safety.")
            elif rewrite_res.is_already_respectful:
                st.success("✅ The original comment is already direct, respectful, and constructive.")

        # Display Candidates
        if rewrite_res.has_rewrites and rewrite_res.candidates:
            st.markdown("---")
            st.markdown("### 🌟 Suggested Constructive Alternatives")
            st.markdown("<p style='color:#888;'>Choose the phrasing that best fits your conversation tone. You can copy it with one click or keep your original message unchanged.</p>", unsafe_allow_html=True)

            for idx, cand in enumerate(rewrite_res.candidates, 1):
                with st.container():
                    st.markdown(f"""
                    <div class='rewrite-box'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
                            <span style='font-size:0.8rem; font-weight:700; color:#68D391;'>Option {idx} · {cand.transformation_type.replace('_', ' ').title()}</span>
                            <span style='font-size:0.75rem; color:#888;'>Confidence: {int(cand.confidence * 100)}%</span>
                        </div>
                        <p style='font-size:1.1rem; font-weight:600; color:#fff; margin:4px 0 10px 0;'>"{cand.suggested_text}"</p>
                        <p style='font-size:0.82rem; color:#a0aec0; margin:0;'><b>Why this works:</b> {cand.explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.code(cand.suggested_text, language="markdown")

        # Retain original action
        st.markdown("---")
        with st.expander("📌 Option to Retain Original Comment Unchanged", expanded=False):
            st.markdown("If you feel the directness or tone of your original comment is necessary in this context, you can keep it as is:")
            st.code(comment_input.strip(), language="markdown")

        # Mandatory Disclaimer
        st.markdown(
            f"<div class='disclaimer-box'><b>Disclaimer:</b> {rewrite_res.disclaimer}</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

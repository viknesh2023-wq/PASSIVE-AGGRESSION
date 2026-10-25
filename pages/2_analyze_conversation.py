"""
pages/2_analyze_conversation.py — Analyse a multi-message conversation thread.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Analyze Conversation — PA Detector 2.0",
    page_icon="🧵",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
.msg-card { background: #111; border: 1px solid #222; border-radius: 8px; padding: 0.9rem 1.2rem; margin: 0.4rem 0; }
.msg-card-selected { background: #111; border: 1px solid #E53E3E; border-radius: 8px; padding: 0.9rem 1.2rem; margin: 0.4rem 0; }
.badge-pa  { background: #3B1212; color: #FC8181; border: 1px solid #E53E3E; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; font-weight: 700; }
.badge-npa { background: #0D2D17; color: #68D391; border: 1px solid #38A169; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; font-weight: 700; }
.badge-amb { background: #2D2010; color: #F6AD55; border: 1px solid #D69E2E; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; font-weight: 700; }
.badge-thr { background: #3B1212; color: #FC8181; border: 1px solid #E53E3E; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }
.badge-ok  { background: #0D2D17; color: #68D391; border: 1px solid #38A169; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }
.muted { color: #666; font-size: 0.82rem; }
.disc-box { background: #0d0d0d; border: 1px solid #333; border-left: 3px solid #D69E2E; border-radius: 6px; padding: 0.7rem 1rem; font-size: 0.83rem; color: #888; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def _load():
    from inference.pa_inference import _load_models, get_model_info
    _load_models()
    return get_model_info()


def _badge(label: str) -> str:
    cls = {
        "passive_aggressive":     "badge-pa",
        "non_passive_aggressive": "badge-npa",
        "ambiguous":              "badge-amb",
        "direct_threat":          "badge-thr",
        "implied_threat":         "badge-thr",
        "no_threat":              "badge-ok",
    }.get(label, "badge-amb")
    return f"<span class='{cls}'>{label.replace('_',' ').title()}</span>"


def _sidebar(model_info):
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
    _sidebar(model_info)

    st.markdown("# 🧵 Analyze Conversation")
    st.markdown(
        "<p style='color:#888;'>Build a conversation thread and analyse any message within its context.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='disc-box'>ℹ️ Context is passed to the model as a prepended string. "
        "The model processes the <b>target message</b> — it does not perform multi-turn reasoning. "
        "Providing parent context may improve accuracy for replies.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Thread Builder ─────────────────────────────────────────────────────────
    # ── Social Media Conversation Presets ──────────────────────────────────────
    st.markdown("#### 📱 Social-Media Thread Presets (Click to load)")
    st.markdown("<p class='muted' style='margin-top:-0.5rem;'>Click any social-media thread preset to auto-populate the original post and multi-turn replies:</p>", unsafe_allow_html=True)

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        if st.button("🔴 Reddit: Debate & Downvote Dogpile", key="t_reddit_debate", use_container_width=True, help="Unpopular opinion gaming thread with snark and passive escalation"):
            st.session_state["orig_post"] = "Unpopular opinion: the new game update completely ruined competitive balance."
            st.session_state.conv_messages = [
                {"sender": "CasualGamer", "text": "Skill issue honestly. The new patch is great if you actually adapt.", "id": 0},
                {"sender": "RankedTryhard", "text": "Tell me you get carried in ranked without telling me you get carried.", "id": 1},
                {"sender": "SaltyVeteran", "text": "Must be nice playing in bronze where patch balance doesn't matter. Glad you're having fun though.", "id": 2},
                {"sender": "OP", "text": "Thanks for the insightful feedback, really shows the maturity of this community.", "id": 3},
            ]
            st.rerun()

        if st.button("📸 Instagram / TikTok: Creator Drama", key="t_insta_drama", use_container_width=True, help="Vlog sponsor drama with passive-aggressive subtext"):
            st.session_state["orig_post"] = "Quick vlog showing my morning routine and brand partnerships!"
            st.session_state.conv_messages = [
                {"sender": "BestieFan", "text": "So happy for all your recent success bestie! You deserve it!", "id": 0},
                {"sender": "SkepticalViewer", "text": "Must be nice getting brand deals when you don't even disclose the sponsorships properly.", "id": 1},
                {"sender": "BestieFan", "text": "She literally put #ad in the caption, why are you always spreading negativity?", "id": 2},
                {"sender": "SkepticalViewer", "text": "Just asking questions. Clearly hit a nerve though.", "id": 3},
            ]
            st.rerun()

    with t_col2:
        if st.button("🐦 Twitter/X: Creator Announcement & Ratio", key="t_twitter_ratio", use_container_width=True, help="Indie game launch ratio thread with snarky replies"):
            st.session_state["orig_post"] = "Thrilled to announce our indie studio's first title after 3 years of hard work!"
            st.session_state.conv_messages = [
                {"sender": "IndieSupporter", "text": "Looks amazing! Wishlisted immediately on Steam.", "id": 0},
                {"sender": "CriticDave", "text": "Interesting how this looks identical to that other indie game from 2022. But sure, 'original'.", "id": 1},
                {"sender": "DevLead", "text": "We drew inspiration from classic arcade games, everything was built from scratch.", "id": 2},
                {"sender": "CriticDave", "text": "Whatever helps you sleep at night. I'm sure the copyright lawyers will love it.", "id": 3},
            ]
            st.rerun()

        if st.button("🎮 Discord / Gaming: LFG Grudge", key="t_discord_lfg", use_container_width=True, help="Discord gaming tournament team conflict"):
            st.session_state["orig_post"] = "LFG need 1 support for tournament tonight, must have mic and be communicative."
            st.session_state.conv_messages = [
                {"sender": "SupportMain", "text": "I can fill, played support all last season.", "id": 0},
                {"sender": "TeamCaptain", "text": "We actually want to win tonight, so maybe sit this one out.", "id": 1},
                {"sender": "SupportMain", "text": "Cool. Good luck finding someone else who will tolerate your attitude.", "id": 2},
                {"sender": "TeamCaptain", "text": "Nobody asked you to stay in this server anyway.", "id": 3},
            ]
            st.rerun()

    st.markdown("---")

    # ── Thread Setup ─────────────────────────────────────────────────────────
    st.markdown("### Thread Setup")

    col1, col2 = st.columns([2, 1])
    with col1:
        if "orig_post" not in st.session_state:
            st.session_state["orig_post"] = ""
        original_post = st.text_area(
            "Original post (optional)",
            height=80,
            placeholder="The original post, tweet, or announcement that started this thread…",
            key="orig_post",
        )

    with col2:
        sensitivity = st.selectbox(
            "Sensitivity",
            ["balanced", "strict", "lenient"],
            index=0,
            key="conv_sensitivity",
            help="Strict = more PA detections; Lenient = fewer",
        )
        context = "thread"

    st.markdown("### Messages in Thread")
    st.markdown("<p class='muted'>Add messages in order. You'll select which one to analyse.</p>", unsafe_allow_html=True)

    # ── Dynamic message list ───────────────────────────────────────────────────
    if "conv_messages" not in st.session_state:
        st.session_state.conv_messages = [
            {"sender": "User A", "text": "", "id": 0},
        ]

    col_add, col_clear = st.columns([1, 5])
    with col_add:
        if st.button("➕ Add message"):
            new_id = max(m["id"] for m in st.session_state.conv_messages) + 1
            st.session_state.conv_messages.append({"sender": "", "text": "", "id": new_id})
            st.rerun()
    with col_clear:
        if st.button("🗑 Clear thread"):
            st.session_state["orig_post"] = ""
            st.session_state.conv_messages = [{"sender": "User A", "text": "", "id": 0}]
            st.rerun()

    # Message input rows
    msgs_to_delete = []
    for i, msg in enumerate(st.session_state.conv_messages):
        with st.container():
            c1, c2, c3, c4 = st.columns([1, 4, 0.5, 0.5])
            with c1:
                sender = st.text_input(
                    "Sender",
                    value=msg.get("sender", ""),
                    key=f"sender_{msg['id']}",
                    placeholder="Username",
                    label_visibility="collapsed",
                )
                st.session_state.conv_messages[i]["sender"] = sender
            with c2:
                text = st.text_area(
                    "Message",
                    value=msg.get("text", ""),
                    height=60,
                    key=f"text_{msg['id']}",
                    placeholder=f"Message {i+1}…",
                    label_visibility="collapsed",
                )
                st.session_state.conv_messages[i]["text"] = text
            with c3:
                st.markdown(f"<p style='padding-top:1.8rem;color:#555;'>#{i+1}</p>", unsafe_allow_html=True)
            with c4:
                if len(st.session_state.conv_messages) > 1:
                    if st.button("✕", key=f"del_{msg['id']}"):
                        msgs_to_delete.append(msg["id"])

    if msgs_to_delete:
        st.session_state.conv_messages = [
            m for m in st.session_state.conv_messages
            if m["id"] not in msgs_to_delete
        ]
        st.rerun()

    st.markdown("---")

    # ── Target selection & analysis ────────────────────────────────────────────
    valid_msgs = [
        m for m in st.session_state.conv_messages
        if m.get("text", "").strip()
    ]

    if len(valid_msgs) == 0:
        st.info("Add at least one message with text to analyse.")
        return

    msg_options = [
        f"#{i+1} [{m.get('sender','?')}]: {m.get('text','')[:60]}{'…' if len(m.get('text',''))>60 else ''}"
        for i, m in enumerate(valid_msgs)
    ]

    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        target_idx = st.selectbox(
            "Select target message to analyse",
            range(len(msg_options)),
            format_func=lambda i: msg_options[i],
            key="target_msg_idx",
        )
    with col_btn:
        st.markdown("<div style='padding-top:1.5rem;'></div>", unsafe_allow_html=True)
        run_all = st.checkbox("Analyse all messages", key="run_all")

    analyse_btn = st.button("🔍 Analyse", type="primary", key="conv_analyse")

    if analyse_btn:
        from inference.pa_inference import analyze

        if run_all:
            _analyse_all(valid_msgs, original_post, context, sensitivity)
        else:
            target = valid_msgs[target_idx]
            parent = None
            if target_idx > 0:
                parent = valid_msgs[target_idx - 1].get("text", "")

            with st.spinner(f"Analysing message from {target.get('sender','?')}…"):
                result = analyze(
                    text=target["text"],
                    context=context,
                    sensitivity=sensitivity,
                    post_text=original_post or None,
                    parent_comment=parent,
                )

            if "error" in result:
                st.error(result["error"])
                return

            # Save to history
            try:
                from inference.history_db import save as save_h
                save_h(result["analysis_id"], target["text"], context, sensitivity, result)
            except Exception:
                pass

            _render_single_result(target, result)


def _analyse_all(msgs, post_text, context, sensitivity):
    """Analyse every message in the thread and show a table."""
    from inference.pa_inference import analyze
    import pandas as pd

    st.markdown("---")
    st.markdown("### Thread Analysis Results")

    rows = []
    for i, msg in enumerate(msgs):
        parent_text = msgs[i - 1].get("text") if i > 0 else None
        with st.spinner(f"Analysing #{i+1}…"):
            result = analyze(
                text=msg["text"],
                context=context,
                sensitivity=sensitivity,
                post_text=post_text or None,
                parent_comment=parent_text,
            )
        ml      = result.get("ml_result", {})
        rule    = result.get("rule_result", {})
        comb    = result.get("combined", {})
        rows.append({
            "#":          i + 1,
            "Sender":     msg.get("sender", "?"),
            "Snippet":    msg.get("text", "")[:60] + ("…" if len(msg.get("text","")) > 60 else ""),
            "ML Label":   ml.get("label", "?"),
            "Rule Label": rule.get("pa_label", "?"),
            "Threat":     comb.get("threat", "?"),
            "PA Score":   ml.get("pa_score", 0),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Highlight PA messages
    pa_rows = [r for r in rows if r["ML Label"] == "passive_aggressive"]
    if pa_rows:
        st.markdown(f"**{len(pa_rows)} message(s) flagged as passive-aggressive by ML:**")
        for r in pa_rows:
            st.markdown(
                f"<div style='background:#3B1212;border:1px solid #E53E3E;border-radius:6px;padding:0.5rem 1rem;margin:3px 0;'>"
                f"<b style='color:#FC8181;'>#{r['#']} {r['Sender']}</b>: {r['Snippet']}</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='disc-box'>ℹ️ Each message is analysed independently with its preceding message as parent context. "
        "No multi-turn reasoning is performed.</div>",
        unsafe_allow_html=True,
    )


def _render_single_result(target: dict, result: dict):
    """Render result for a single selected message."""
    from pages._shared_result import render_result_panel
    render_result_panel(result, target["text"], target.get("sender", "?"))


# ── Shared result renderer (inline to avoid import issues) ─────────────────────
# We define the rendering directly here as a fallback

def _render_single_result(target: dict, result: dict):
    st.markdown("---")
    st.markdown(f"### Result: Message from **{target.get('sender','?')}**")
    st.markdown(
        f"<blockquote style='border-left:3px solid #333;padding:0.5rem 1rem;color:#aaa;'>"
        f"{target.get('text','')[:300]}</blockquote>",
        unsafe_allow_html=True,
    )

    ml     = result.get("ml_result", {})
    rule   = result.get("rule_result", {})
    threat = result.get("threat_result", {})
    comb   = result.get("combined", {})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**ML Result**")
        st.markdown(_badge(ml.get("label", "non_passive_aggressive")), unsafe_allow_html=True)
        st.metric("PA Score", f"{ml.get('pa_score', 0):.1f}")
    with c2:
        st.markdown("**Rule-Based**")
        st.markdown(_badge(rule.get("pa_label", "non_passive_aggressive")), unsafe_allow_html=True)
        tact = rule.get("tactics", [])
        st.caption(", ".join(t.replace("_"," ") for t in tact[:3]) if tact else "No tactics")
    with c3:
        st.markdown("**Threat**")
        threat_lbl = comb.get("threat", threat.get("label", "no_threat"))
        st.markdown(_badge(threat_lbl), unsafe_allow_html=True)

    signals = rule.get("signals", [])
    if signals:
        with st.expander(f"Rule signals ({len(signals)})"):
            for sig in signals[:6]:
                st.markdown(
                    f"- **\"{sig.get('phrase','')}\"** `{sig.get('tactic','').replace('_',' ')}` "
                    f"— {sig.get('reason','')}",
                )

    flags = comb.get("flags", [])
    for flag in flags:
        if "⚠️" in flag or "Threat" in flag:
            st.warning(flag)
        else:
            st.info(flag)


if __name__ == "__main__":
    main()

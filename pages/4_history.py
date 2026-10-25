"""
pages/3_history.py — View previously saved analyses.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="History — PA Detector 2.0",
    page_icon="📋",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] { background-color: #0a0a0a; border-right: 1px solid #222; }
.badge-pa  { background: #3B1212; color: #FC8181; border: 1px solid #E53E3E; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }
.badge-npa { background: #0D2D17; color: #68D391; border: 1px solid #38A169; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }
.badge-amb { background: #2D2010; color: #F6AD55; border: 1px solid #D69E2E; border-radius: 4px; padding: 2px 8px; font-size: 0.8rem; }
.muted { color: #666; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


def _sidebar():
    with st.sidebar:
        st.markdown("## 🎯 PA Detector 2.0")
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

    st.markdown("# 📋 Analysis History")
    st.markdown(
        "<p style='color:#888;'>Previously saved analyses from this session and earlier sessions.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    try:
        from inference.history_db import get_all, delete_all, get_stats
        rows     = get_all(limit=200)
        stats    = get_stats()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    total = stats.get("total", 0)

    if total == 0:
        st.info("No analyses saved yet. Use **Analyze Comment** or **Analyze Conversation** to get started.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────────
    by_label  = stats.get("by_label", {})
    by_threat = stats.get("by_threat", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total", total)
    m2.metric("Passive-Aggressive", by_label.get("passive_aggressive", 0))
    m3.metric("Non-PA", by_label.get("non_passive_aggressive", 0))
    m4.metric("Ambiguous", by_label.get("ambiguous", 0))
    m5.metric("Threats Flagged",
              by_threat.get("direct_threat", 0) + by_threat.get("implied_threat", 0))

    st.markdown("---")

    if "history_search_kw" not in st.session_state:
        st.session_state["history_search_kw"] = ""

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        label_filter = st.selectbox(
            "Filter by ML label",
            ["All", "passive_aggressive", "non_passive_aggressive", "ambiguous"],
        )
    with col_f2:
        threat_filter = st.selectbox(
            "Filter by threat",
            ["All", "no_threat", "direct_threat", "implied_threat", "ambiguous"],
        )
    with col_f3:
        search = st.text_input("Search text snippet", placeholder="Enter keyword…", key="history_search_kw")

    # Quick filter presets
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        if st.button("Reset / All", key="kw_all", use_container_width=True):
            st.session_state["history_search_kw"] = ""
            st.rerun()
    with k2:
        if st.button("snark / read", key="kw_read", use_container_width=True):
            st.session_state["history_search_kw"] = "read"
            st.rerun()
    with k3:
        if st.button("must be nice", key="kw_nice", use_container_width=True):
            st.session_state["history_search_kw"] = "must be nice"
            st.rerun()
    with k4:
        if st.button("downvote / take", key="kw_downvote", use_container_width=True):
            st.session_state["history_search_kw"] = "take"
            st.rerun()
    with k5:
        if st.button("whatever", key="kw_whatever", use_container_width=True):
            st.session_state["history_search_kw"] = "whatever"
            st.rerun()
    with k6:
        if st.button("threat / safety", key="kw_threat", use_container_width=True):
            st.session_state["history_search_kw"] = "live"
            st.rerun()

    # ── Build table ────────────────────────────────────────────────────────────
    table_rows = []
    for r in rows:
        ml_label    = r.get("ml_label", "?")
        rule_label  = r.get("rule_label", "?")
        threat      = r.get("threat_label", "?")
        snippet     = r.get("text_snippet", "?")
        ts          = r.get("timestamp", "")[:19].replace("T", " ")
        score       = r.get("pa_score", 0)
        context     = r.get("context", "?")
        model_v     = r.get("model_version", "?")
        aid         = r.get("id", "?")

        # Apply filters
        if label_filter != "All" and ml_label != label_filter:
            continue
        if threat_filter != "All" and threat != threat_filter:
            continue
        if search and search.lower() not in snippet.lower():
            continue

        table_rows.append({
            "ID":           aid,
            "Timestamp":    ts,
            "Snippet":      snippet,
            "ML Label":     ml_label,
            "Rule Label":   rule_label,
            "Threat":       threat,
            "PA Score":     round(score, 1),
            "Context":      context,
            "Model":        model_v,
        })

    if not table_rows:
        st.info("No records match the current filters.")
        return

    st.markdown(f"**Showing {len(table_rows)} of {total} records**")

    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Detail view ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### View Details")
    selected_id = st.selectbox(
        "Select an analysis ID to view full details",
        options=[""] + [r["ID"] for r in table_rows],
        format_func=lambda x: x if x else "— select —",
    )

    if selected_id:
        try:
            from inference.history_db import get_one
            full = get_one(selected_id)
            if full:
                ml     = full.get("ml_result", {})
                rule   = full.get("rule_result", {})
                comb   = full.get("combined", {})
                threat = full.get("threat_result", {})

                st.markdown(f"**Analysis `{selected_id}`**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**ML Label:** `{ml.get('label','?')}`")
                    st.markdown(f"**PA Score:** {ml.get('pa_score',0):.1f}")
                with c2:
                    st.markdown(f"**Rule Label:** `{rule.get('pa_label','?')}`")
                    tactics = rule.get("tactics", [])
                    st.markdown(f"**Tactics:** {', '.join(t.replace('_',' ') for t in tactics[:3]) or 'None'}")
                with c3:
                    st.markdown(f"**Threat:** `{comb.get('threat','?')}`")

                explanation = comb.get("explanation", "")
                if explanation:
                    st.markdown(f"> {explanation}")

                signals = rule.get("signals", [])
                if signals:
                    with st.expander(f"Rule signals ({len(signals)})"):
                        for sig in signals[:8]:
                            st.markdown(
                                f"- **\"{sig.get('phrase','')}\"** `{sig.get('tactic','').replace('_',' ')}` — {sig.get('reason','')}"
                            )

                with st.expander("Full JSON result"):
                    st.json(full)
            else:
                st.warning(f"Record `{selected_id}` not found.")
        except Exception as e:
            st.error(f"Error loading record: {e}")

    # ── Export & Clear ────────────────────────────────────────────────────────
    st.markdown("---")
    col_exp, col_del = st.columns([2, 1])
    with col_exp:
        if st.button("📥 Export visible rows as CSV"):
            df_export = pd.DataFrame(table_rows)
            csv_str   = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_str,
                file_name="pa_history_export.csv",
                mime="text/csv",
            )

    with col_del:
        if st.button("🗑 Clear all history", type="secondary"):
            count = delete_all()
            st.success(f"Deleted {count} records.")
            st.rerun()


if __name__ == "__main__":
    main()

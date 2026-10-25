# System Limitations and Ethical Safeguards

**Project:** PassiveAggression Detector  
**Document:** Limitations & Deployment Boundaries  
**Version:** 2.0  

---

## 1. Technical & Modeling Limitations

1. **Context Boundary:** Shorthand statements (e.g., *"Fine."*, *"Whatever."*, *"If you say so."*) are intrinsically ambiguous in isolation. In casual texting, they may signify simple agreement or neutral conciseness, while in workplace email chains they frequently denote hostility or dismissive resignation.
2. **Domain Adaptation:** The classifier is optimized for English professional email, Slack/Teams workplace chat, and general social text. Performance may degrade on niche domains, heavy technical jargon, or non-standard regional dialects.
3. **Subtle Irony Edge Cases:** Sarcasm that relies entirely on unstated shared context between intimates without any textual marker cannot be detected from single-sentence surface text alone.
4. **Latency Trade-offs:**
   - **TF-IDF + Linear SVM:** 0.03 ms per sample (ideal for real-time keystroke suggestions).
   - **XLM-RoBERTa Transformer:** 12–15 ms per sample on GPU / 50 ms on CPU (ideal for batch asynchronous scanning).

---

## 2. Ethical Safeguards & User Agency

1. **Not a Psychological Profiler:** The 0–100 score represents a **textual indirect hostility index**, never a clinical diagnosis or assessment of an individual's character or mental state.
2. **Deterministic Rewrites Require User Confirmation:** The calmer rewrite engine produces suggestions deterministically without hallucinations; however, the user must explicitly inspect and confirm any replacement text before sending.
3. **Privacy Preserving by Default:**
   - The backend runs 100% locally on user hardware without transmitting text to third-party paid APIs.
   - The local SQLite database stores only truncated 100-character snippets for history logging rather than full communication archives.
4. **Uncertainty Flagging:** The system explicitly signals when predictions are low-confidence ($H_{norm} > 0.75$) via the `is_uncertain: true` flag, advising users that manual human judgment is necessary.

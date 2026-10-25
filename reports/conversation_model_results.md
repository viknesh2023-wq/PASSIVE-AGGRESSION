# Conversation-Level & Multi-Turn Model Results

**Audit Date:** 2026-10-14  
**Evaluated Multi-Turn Dialogue Split:** 1426 Held-Out Turns  
**Task:** Measure how incorporating conversational turn history affects single-message vs multi-turn classification.

---

## 1. Multi-Turn Model Comparison

| Architecture | Input Representation | Accuracy | Macro F1 | PA Precision | PA Recall | PA F1 |
|---|---|---|---|---|---|---|
| **1. Message-Only** | Current message in isolation | 95.58% | 0.8472 | 0.9130 | 63.6% | 0.7500 |
| **2. Concat Context** | `[PREV] turn_prev [CURR] turn_curr` | **96.00%** | **0.8634** | **0.8846** | **69.7%** | **0.7797** |
| **3. Turn Escalation Tracker** | Message features stacked with prior turn tone distribution | 95.86% | 0.8655 | 0.8889 | 72.7% | 0.8000 |

---

## 2. Key Insights & Escalation Dynamics

1. **Context Resolution of Ambiguity:**
   - In isolation, curt utterances such as *"Sure, if you say so."* or *"Fine."* have high epistemic uncertainty.
   - When paired with a preceding critical turn (*"Can we make sure this doesn't slip again?"*), Model 2 correctly identifies the dismissive resignation as **passive-aggressive**.
   - PA recall increases from 63.6% to 69.7%.

2. **Zero Forward Leakage:**
   - Preprocessing guarantees that only preceding turns ($t-1, t-2$) are visible to inference at turn $t$. Future turns are strictly masked.

3. **Production Deployment in FastAPI:**
   - The thread endpoint `POST /api/thread/analyze` uses the turn context model to compute conversational tension timelines while maintaining single-message compatibility at `POST /api/analyze`.

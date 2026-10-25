# Annotation Guidelines and Tactic Taxonomy

**Project:** PassiveAggression Detector  
**Document:** Annotation Guide & Taxonomy Reference  
**Version:** 2.0  

---

## 1. Primary Classification Labels

### 1. `neutral`
Text that is factually objective, polite, constructive, or represents non-hostile conversational banter.
- Direct disagreement without contempt or indirect malice is categorized as `neutral`.
- Playful irony or lighthearted sarcasm with friendly intent is categorized as `neutral`.
- *Example:* "I respectfully disagree with the proposed database schema; benchmark tests indicate higher latency."

### 2. `passive_aggressive`
Hostility, resentment, or dissatisfaction conveyed through indirect phrasing, faux-politeness, guilt, or deliberate avoidance.
- Discrepancy between surface politeness and covert resentment.
- Subtly shifting blame to the recipient while avoiding direct accountability.
- *Example:* "As per my last email, the deadline was Friday. I assume you were occupied."

### 3. `overt_hostile`
Explicit hostility, direct verbal abuse, severe insults, harassment, or intimidation.
- Surface text is openly aggressive with no attempt to disguise negativity.
- *Example:* "You are an incompetent idiot and everybody on this engineering team despises working with you."

---

## 2. Six Passive-Aggressive Tactics

| Tactic | Definition | Diagnostic Linguistic Markers | Example |
|---|---|---|---|
| **`mock_politeness`** | Professional or polite framing weaponized to deliver a condescending or dismissive barb. | *"With all due respect"*, *"Thanks in advance for actually"*, *"Kindly refer to"* | *"Thanks in advance for finally reading the documentation."* |
| **`blame_shifting`** | Indirectly redirecting responsibility or fault onto the reader while deflecting accountability. | *"Per my last email"*, *"As I already mentioned"*, *"I assumed you knew"* | *"As per my last email, the instructions were already circulated."* |
| **`backhanded_compliment`** | A statement disguised as praise that contains an implicit insult or condescending qualification. | *"Surprisingly competent"*, *"Better than usual"*, *"At least you tried"* | *"Your presentation was surprisingly coherent given how little you prepared."* |
| **`guilt_tripping`** | Implying the recipient is neglectful, selfish, or placing an unfair burden on the speaker. | *"Don't worry about me"*, *"Must be nice to have so much free time"*, *"I'll just work alone"* | *"Go ahead and log off, I'll just finish your slides on my personal weekend."* |
| **`curt_dismissal`** | Abrupt shorthand indicating disengagement, suppressed irritation, or refusal to collaborate. | *"Fine."*, *"Whatever you say."*, *"K."*, *"Noted."*, *"If you say so."* | *"Fine. Do whatever you want, you always do anyway."* |
| **`sarcasm`** | Bitter or cutting irony intended to mock, disparage, or express indirect hostility. | *"Oh brilliant"*, *"What a shock"*, *"Because clearly you know best"* | *"Oh wonderful, another two-hour sync that could have been a three-line Slack post."* |

---

## 3. Contrastive Pair Decision Guidelines

Annotators must evaluate communicative intent and conversational context:

| Potentially Ambiguous Text | Neutral Interpretation | Passive-Aggressive Interpretation |
|---|---|---|
| *"Per my last email, the report is attached."* | **Neutral:** Neutral informational reference to previous correspondence. | N/A (Lacks hostile or condescending qualification). |
| *"Per my last email, which you appear to have forgotten."* | N/A | **Passive-Aggressive:** Explicitly attacks recipient's attentiveness. |
| *"Fine, that works for my schedule."* | **Neutral:** Sincere agreement and confirmation of time. | N/A |
| *"Fine. Do whatever you want."* | N/A | **Passive-Aggressive:** Sulky resignation, hostility via curt dismissal. |
| *"Thanks in advance for your assistance."* | **Neutral:** Sincere gratitude for upcoming teamwork. | N/A |
| *"Thanks in advance for actually doing your job."* | N/A | **Passive-Aggressive:** Weaponized politeness insulting recipient's work ethic. |

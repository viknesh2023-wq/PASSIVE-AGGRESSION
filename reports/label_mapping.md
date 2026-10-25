# Task-Specific Label Mapping Specification

**Target System:** PassiveAggression Detector 2.0  
**Specification:** Explicit semantic separation between primary covert hostility (PA), overt hostility, playful sarcasm, neutral sentiment, and independent threats.

---

## 1. Primary Classification Mapping Rules

| Source Dataset | Original Label | Primary Task Mapping | Threat Mapping | Rationale |
|---|---|---|---|---|
| `tweeteval_sentiment` | `1` (Neutral) | `non_passive_aggressive` | `no_threat` | Sincere, objective conversational statements. |
| `sarc_reddit_regular` | `0` (Regular) | `non_passive_aggressive` | `no_threat` | Ordinary everyday social media dialogue. |
| `sarc_reddit_sarcasm` | `1` (Sarcastic) | `non_passive_aggressive` | `no_threat` | Non-hostile, playful irony/sarcasm (hard negative). |
| `genz_synthetic_sarcasm` | `banter` | `non_passive_aggressive` | `no_threat` | Informal generational banter; non-toxic. |
| `goemotions_hostile` | `anger/annoyance` | `non_passive_aggressive` | `no_threat` | Direct overt aggression/insults; NOT covert/indirect PA. |
| `sarc_mined_pa` | `mined_cue` | `passive_aggressive` | `no_threat` | High-precision covert hostility cues (*"Must be nice"*, *"As per my"*). |
| `curated_pa_expansion` | `tactic_templates`| `passive_aggressive` | `no_threat` | Syntactic expansions across 6 indirect tactics. |
| `contrastive_hard_negatives`| `contrastive`| `non_passive_aggressive`| `no_threat` | Minimal pairs containing polite/sincere phrasing. |
| `gold_annotation_template` | Multi-annotator | Verified Gold | Verified Gold | Human ground truth reference anchors. |
| `synthetic_pa_dataset` | 3 Canonical Classes| Identity Mapping | Independent Task | Dedicated 9,000-record primary corpus. |

---

## 2. Seven Passive-Aggression Tactic Subtypes
1. `fake_politeness`: Mock deference, weaponized politeness (*"As per my last email"*, *"Bless your heart"*).
2. `indirect_criticism`: Veiled observations (*"I'm just saying"*, *"Interesting how"*).
3. `backhanded_compliment`: Double-edged praise (*"Surprisingly good for you"*, *"Better late than never"*).
4. `implied_blame`: Counterfactual guilt (*"If someone had actually checked"*, *"I already told you"*).
5. `sarcastic_agreement`: Feigned compliance (*"If you say so"*, *"Great, so now..."*).
6. `dismissiveness`: Curt disengagement (*"Whatever you say"*, *"Fine."*).
7. `other_or_unclear`: Ambiguous or unclassified indirect signals.

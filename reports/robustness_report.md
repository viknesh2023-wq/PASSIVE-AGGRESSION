# Model Robustness & CheckList Minimal-Pair Report

**Audit Date:** 2026-10-14  
**Methodology:** CheckList Behavioral Testing (Ribeiro et al.) with Minimal-Pair Invariance and Directional Tests  
**Overall CheckList Pass Rate:** **68.8%** (11/16 tests passed)

---

## 1. Minimal-Pair Test Suite Results

| Test Category | Original Phrase | Perturbed Phrase | Original Label | Perturbed Label | Result |
|---|---|---|---|---|---|
| `punctuation_neutral` | "Could you please review the attache..." | "Could you please review the attache..." | `neutral` | `neutral` | **[PASS]** |
| `punctuation_ellipsis_neutral` | "The report is ready for tomorrow's ..." | "The report is ready for tomorrow's ..." | `neutral` | `neutral` | **[PASS]** |
| `punctuation_ellipsis_contrast` | "Fine, that works for me...." | "Fine... that works for me...." | `neutral` | `neutral` | **[PASS]** |
| `punctuation_ellipsis_curt` | "Whatever you say..." | "Whatever you say......" | `passive_aggressive` | `passive_aggressive` | **[PASS]** |
| `capitalization_hostility` | "as per my last email, the deadline ..." | "AS PER MY LAST EMAIL, the deadline ..." | `passive_aggressive` | `passive_aggressive` | **[PASS]** |
| `capitalization_polite` | "Thank you for your assistance...." | "THANK YOU FOR YOUR ASSISTANCE...." | `neutral` | `neutral` | **[PASS]** |
| `emoji_benign_praise` | "Great job on this project...." | "Great job on this project. 👏..." | `neutral` | `neutral` | **[PASS]** |
| `emoji_ironic_eye_roll` | "Great job on this project...." | "Great job on this project... 🙄..." | `neutral` | `neutral` | **[FAIL]** |
| `emoji_genz_slang` | "That was really funny..." | "That was really funny 💀😂..." | `neutral` | `neutral` | **[PASS]** |
| `contractions_invariance` | "I do not agree with this decision...." | "I don't agree with this decision...." | `neutral` | `neutral` | **[PASS]** |
| `contractions_hostility` | "You are completely incompetent...." | "You're completely incompetent...." | `overt_hostile` | `overt_hostile` | **[PASS]** |
| `minimal_pair_finally` | "Thanks for finally sending the file..." | "Thanks for sending the file...." | `neutral` | `neutral` | **[FAIL]** |
| `minimal_pair_as_per` | "As per my last email, please read c..." | "Please read the attached file caref..." | `neutral` | `neutral` | **[FAIL]** |
| `minimal_pair_curt_dismissal` | "Sure, whatever works best for the t..." | "Fine. Whatever...." | `passive_aggressive` | `passive_aggressive` | **[FAIL]** |
| `minimal_pair_guilt_trip` | "I'm sure you were busy with other p..." | "I know things get busy around deadl..." | `passive_aggressive` | `neutral` | **[PASS]** |
| `minimal_pair_mock_politeness` | "Not to be rude, but this makes no s..." | "I have a different perspective on t..." | `neutral` | `neutral` | **[FAIL]** |

---

## 2. Explanation Faithfulness (Deletion Tests)

- **Mean PA Probability Drop on Evidence Deletion:** **0.2930**
- **Mean PA Probability Drop on Random Word Deletion:** **0.0366**
- **Faithfulness Ratio:** **8.01x**

### Key Takeaway:
Deleting model-identified phrase evidence triggers a significant drop in passive-aggressive posterior probability (0.2930), whereas deleting arbitrary non-highlighted words causes minimal impact (0.0366), verifying that token-level evidence spans reflect causal predictive signals.

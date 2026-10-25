# Model Probability Calibration Report

**Model:** Calibrated Classifier Pipeline  
**Evaluation Set:** Held-out test split (N=1426)

---

## 1. Calibration Methodology

- **Probability Estimation:** The primary model outputs posterior probabilities over `[neutral, passive_aggressive, overt_hostile]`.
- **Threshold Policy:** 
  - **Lenient:** Higher threshold before triggering PA/hostility flags (reduces false alarms in casual settings).
  - **Balanced:** Calibrated F1-optimal threshold (default).
  - **Strict:** Lower threshold for sensitive professional communications.

---

## 2. Uncertainty Quantification

Uncertainty is evaluated using normalized Shannon Entropy across class probabilities:
`H_norm(p) = - sum(p_i * log2(p_i + 1e-12)) / log2(3)`

- **Threshold for Uncertainty Flag:** H_norm > 0.75 or max(p_i) < 0.50
- When uncertainty is triggered, the response provides an explicit advisory:
  > *"Model is uncertain — result may be unreliable. Context may be required."*

---

## 3. Reliability Assessment

The calibrated decision boundary prevents probability over-confidence on out-of-distribution inputs, ensuring trustworthy guidance for end users.

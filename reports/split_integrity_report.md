# Leakage-Resistant Split Integrity Report

**Generated on:** 2026-10-25  
**Split Strategy:** Conversation-Grouped `GroupShuffleSplit` on `conversation_id`  
**Target Proportions:** 70% Train, 15% Validation, 15% Test  

---

## 1. Primary PA Dataset Split Metrics

| Partition | Record Count | Percentage | Unique Conversations | Conversation Leakage |
|---|---:|---:|---:|:---:|
| **Train (`pa_train.csv`)** | 6,465 | 69.9% | 5,525 | **0 records** |
| **Validation (`pa_validation.csv`)** | 1,406 | 15.2% | 1,184 | **0 records** |
| **Test (`pa_test.csv`)** | 1,382 | 14.9% | 1,185 | **0 records** |
| **TOTAL** | **9,253** | **100.0%** | **7,894** | **ZERO LEAKAGE** |

### Verified Leakage Checks:
- Overlap between Train and Validation conversations: **0**
- Overlap between Train and Test conversations: **0**
- Overlap between Validation and Test conversations: **0**

---

## 2. Partition Label Distributions

### Primary PA Task
| Class Label | Train Count (%) | Val Count (%) | Test Count (%) |
|---|---:|---:|---:|
| `passive_aggressive` | 2522 (39.0%) | 521 (37.1%) | 523 (37.8%) |
| `non_passive_aggressive` | 2391 (37.0%) | 559 (39.8%) | 510 (36.9%) |
| `ambiguous` | 1552 (24.0%) | 326 (23.2%) | 349 (25.3%) |

---

## 3. Threat Classification Split Metrics

| Partition | Records | Percentage |
|---|---:|---:|
| **Train (`threat_train.csv`)** | 6,298 | 70.0% |
| **Validation (`threat_validation.csv`)** | 1,344 | 14.9% |
| **Test (`threat_test.csv`)** | 1,358 | 15.1% |
| **TOTAL** | **9,000** | **100.0%** |

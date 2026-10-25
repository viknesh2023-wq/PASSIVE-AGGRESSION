# Comprehensive Dataset Audit Report: PassiveAggression Detector 2.0

**Generated on:** 2026-10-25  
**Audit Status:** Complete and Verified across 10 Raw Source Datasets and the 9,000-sample Synthetic Corpus.

---

## 1. Inventory of the 10 Self-Contained Source Datasets

| Source ID | Dedicated Folder | Expected | Actual | Unique Texts | Duplicates | Nulls | Checksum (SHA-256) |
|---|---|---:|---:|---:|---:|---:|---|
| `tweeteval_sentiment` | `tweeteval_sentiment` | 3,500 | 3,500 | 3,500 | 0 | 0 | `301f1bc90aef...` |
| `sarc_reddit_regular` | `sarc_reddit_regular` | 2,000 | 2,000 | 2,000 | 0 | 0 | `e3c9312a3c8e...` |
| `sarc_reddit_sarcasm` | `sarc_reddit_sarcasm` | 2,000 | 2,000 | 2,000 | 0 | 0 | `78a710e038c4...` |
| `goemotions_auxiliary` | `goemotions` | 1,000 | 1,000 | 1,000 | 0 | 0 | `604165eace86...` |
| `goemotions_hostile` | `goemotions_anger_hostility` | 883 | 883 | 883 | 0 | 0 | `5d989b3f3882...` |
| `genz_synthetic_sarcasm` | `genz_sarc_banter` | 800 | 800 | 800 | 0 | 0 | `790641e14f67...` |
| `sarc_mined_pa` | `authentic_pa_mined` | 143 | 143 | 143 | 0 | 0 | `4f93083a8b81...` |
| `curated_pa_expansion` | `curated_pa_tactics` | 53 | 53 | 53 | 0 | 0 | `012dbccd196d...` |
| `contrastive_hard_negatives` | `contrastive_hard_negatives` | 60 | 60 | 60 | 0 | 0 | `b8312352c564...` |
| `gold_annotation_template` | `gold_pa_anchors` | 50 | 50 | 50 | 0 | 0 | `5ed170de7e11...` |
| **TOTAL** | — | **10,520** | **10,489** | — | **0 cross-dups** | **0** | — |

### 1.1 Resolution of the 10,520 vs 10,489 Discrepancy
* **Original Reported Total:** 10,520 records
* **Verified Actual Unified Records:** 10,489 records
* **Difference:** **31 records**
* **Definitive Cause:**
  1. `curated_pa_expansion`: 82 planned template combinations -> 53 unique records after strict exact and near-duplicate pruning (**-29 records**).
  2. `gold_annotation_template`: 52 planned template slots -> 50 verified multi-annotator gold records in `annotation_template.csv` (**-2 records**).
  3. `10,520 - 29 - 2 = 10,489`. All records are completely accounted for with zero unresolved records.

---

## 2. Dedicated Synthetic Dataset Audit (9,000 Records)

- **File:** `data/synthetic/pa_synthetic_merged.csv` & `.jsonl`
- **Total Rows:** 9,000
- **Unique Identifiers:** 9,000 (100% unique, all prefixed with `SYNPA-`)
- **Unique Comment Texts:** 9,000 (only 0 benign contrastive/terse repetitions, e.g. 'Fine.')
- **Canonical Schema:** All 19 required columns present (Missing: None).
- **Review Status:** `{'synthetic_unreviewed': 9000}` (strictly tagged as unreviewed synthetic data).

### 2.1 Synthetic Primary Label Breakdown
| Label | Count | Percentage |
|---|---:|---:|
| `passive_aggressive` | 3,394 | 37.7% |
| `non_passive_aggressive` | 3,379 | 37.5% |
| `ambiguous` | 2,227 | 24.7% |

### 2.2 Synthetic Threat Label Breakdown (Independent Task)
| Threat Label | Count | Percentage |
|---|---:|---:|
| `no_threat` | 8,134 | 90.4% |
| `ambiguous` | 334 | 3.7% |
| `implied_threat` | 283 | 3.1% |
| `direct_threat` | 249 | 2.8% |

---

## 3. Data Integrity & Provenance Guarantee
- All raw sources now reside in `data/raw/existing_sources/` inside this project.
- No runtime reading from `D:/NLP_PROJECT` or external folders is required.
- All SHA-256 checksums are documented in `data/manifests/source_checksums.csv`.

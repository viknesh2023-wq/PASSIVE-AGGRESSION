# Source Project Audit Report — GATE 1
## PassiveAggression Detector

**Audit Date:** 2026-10-14  
**Status:** COMPLETE — All findings verified against live filesystem

---

## 1. Environment Summary

| Item | Value |
|------|-------|
| Python | 3.12.10 CPython MSC v.1943 64-bit AMD64 |
| Python executable | C:\Users\Viknesh G\AppData\Local\Programs\Python\Python312\python.exe |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU VRAM total | 6141 MiB |
| GPU VRAM free | 5920 MiB |
| OS | Windows 11 10.0.26200 |
| D: Total | 161.0 GB |
| D: Free | 73.0 GB |

### Installed in Base Python Env

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.14.0 | CPU + CUDA available |
| transformers | 5.17.0 | HuggingFace |
| scikit-learn | 1.9.1 | ML baselines |
| sentence-transformers | 6.1.0 | Embedding models |
| huggingface_hub | 1.33.0 | Model download |
| pandas | 3.0.6 | Data wrangling |
| numpy | 2.5.3 | Numerics |
| lightgbm | 4.7.0 | Tree models |
| xgboost | 3.4.1 | Tree models |
| fastapi | NOT INSTALLED | Installing in GATE 2 |
| uvicorn | NOT INSTALLED | Installing in GATE 2 |
| spaCy | NOT INSTALLED | Installing in GATE 2 |
| nltk/vader | NOT INSTALLED | Installing in GATE 2 |

> internvl_env at D:\NLP_PROJECT\internvl_env has FastAPI, uvicorn, spaCy — confirmed present but we install fresh in base env.

---

## 2. Existing Frontend Audit (d:\python_folder)

**Status: PRESERVED — NOT modified**

| Item | Detail |
|------|--------|
| Framework | React 19.2.8 + Vite 8.3.0 + TypeScript 6.0.2 |
| CSS | Tailwind CSS v4 + custom design tokens |
| State | Zustand 5.0.15 + localStorage |
| Data fetching | @tanstack/react-query 5.104.1 |
| Charts | recharts 3.10.1 |
| Animations | framer-motion 14.0.0 |
| Icons | lucide-react 1.53.0 |
| API base | http://localhost:8000 via VITE_API_BASE_URL |
| Mock mode | VITE_USE_MOCK=true (default — no backend yet) |

### Pages Confirmed Working

| Route | Component | Status |
|-------|-----------|--------|
| / | Analyzer.tsx | Complete + mock engine |
| /thread | Thread.tsx | Complete + mock engine |
| /history | History.tsx | Complete localStorage |
| /insights | Insights.tsx | Complete |
| /how-it-works | HowItWorks.tsx | Complete |

### API Contract Expected by Frontend

```
POST /api/analyze   body: {text, context, sensitivity}
                    -> {id, label, score, tactics[], highlights[], sentences[], rewrites[], model_version, latency_ms}

POST /api/batch     body: {messages: [{sender, text, time?}], context, sensitivity}
                    -> {results: [...AnalyzeResponse + sender]}

POST /api/feedback  body: {id, helpful, corrected_label?}
                    -> void / 200

GET  /api/history   -> AnalyzeResponse[]
GET  /api/health    -> true/false (bool)
```

### TypeScript Label Enums Used in Frontend

```
Label      = 'neutral' | 'passive_aggressive' | 'overt_hostile'
Tactic     = 'sarcasm' | 'backhanded_compliment' | 'guilt_tripping' | 'mock_politeness' | 'blame_shifting' | 'curt_dismissal'
Context    = 'workplace' | 'chat' | 'social'
Sensitivity = 'lenient' | 'balanced' | 'strict'
```

---

## 3. D:\NLP_PROJECT — Read-Only Audit

### 3.1 Top-Level Directories

| Directory | Purpose | Relevance |
|-----------|---------|-----------|
| 06_MMSD2.0 | Multimodal Sarcasm Dataset 2.0 (images+tweets) | Low — image-based |
| backend | Existing FastAPI multimodal sarcasm backend | E — incompatible task |
| DravidianCodeMix | Dravidian language code-mix | E — language mismatch |
| FINAL_DATASETS | Cleaned combined datasets (TEXT_SARCASM, SENTIMENT, EMOTION) | B — auxiliary |
| GenZ-Sarc | 20 batches synthetic Gen-Z sarcasm CSVs | B — hard negatives |
| GoEmotions | 28-class Reddit emotion dataset | B — auxiliary features |
| internvl_env | Python venv with FastAPI + CUDA libs | A — infra reference |
| isarcasm | iSarcasmEval CSVs (tweet_ids only, no text) | F — unusable |
| MEMOTION | Meme image + emotion dataset | E — multimodal |
| models | Trained XLM-R checkpoints | C — encoders |
| sarc | SARC Reddit sarcasm corpus | B — hard negatives |
| sentimix | SentiMix code-mix sentiment | E — language mismatch |
| TweetEval | Twitter 3-class sentiment | B — hard negatives |
| neutral_darav | Synthetic neutral code-mixed | B — neutral examples |

### 3.2 Dataset Details

#### TEXT_SARCASM (FINAL_DATASETS)
- **File:** text_sarcasm_combined.csv
- **Shape:** 1,005,167 rows x 10 columns
- **Columns:** id, text, label, language, source, style, topic, slang_used, abbreviations_used, emoji_used
- **Label dist:** 0=not_sarcastic 499,348 | 1=sarcastic 505,819
- **Sources:** SARC 956,097 | GenZ-SarC 19,984 | Multilingual-Synthetic 19,999 | SarcasmCorpusV2 9,087
- **PA relevance:** Hard-negative text source ONLY — sarcasm != passive aggression
- **License:** SARC research only; GenZ-Sarc synthetic; others need verification

#### SENTIMENT (FINAL_DATASETS)
- **File:** sentiment_combined.csv
- **Shape:** 132,324 rows x 8 columns
- **Label dist:** 0=Negative 25,109 | 1=Neutral 45,405 | 2=Positive 61,810
- **PA relevance:** Hard negatives — negative sentiment does NOT imply PA

#### EMOTION (FINAL_DATASETS)
- **File:** emotion_combined.csv
- **Shape:** 54,263 rows x 8 columns
- **Source:** GoEmotions (Reddit, 28 classes)
- **Top labels:** neutral 16,021 | admiration 3,384 | gratitude 2,378 | annoyance 1,809 | anger 1,265
- **PA relevance:** annoyance + disapproval subset may help overt_hostile detection

#### GoEmotions (raw)
- **Train:** 43,410 | Dev: ~5,426 | Test: ~5,426
- **Format:** TSV — text | label_ids | id
- **PA relevance:** Same as EMOTION above

#### iSarcasm
- **Train:** 3,577 | Test: 907
- **CRITICAL ISSUE:** Only tweet_id column present — NO TEXT
- **Status:** UNUSABLE without Twitter API

#### SARC
- **test-balanced.csv:** 32,332 rows (readable with skip)
- **train-balanced-sarc.csv:** ~240 MB, malformed CSV
- **PA relevance:** Hard-negative mining for sarcasm that is NOT passive-aggressive

#### TweetEval
- **Train:** 45,615 rows
- **Task:** Sentiment 0=negative, 1=neutral, 2=positive
- **PA relevance:** Negative sentiment examples for hard negatives

### 3.3 Trained Models

#### xlmr_text_sarcasm_full (LARGEST, BEST)
- Architecture: XLM-RoBERTa base (FacebookAI/xlm-roberta-base)
- Task: BINARY SARCASM — not PA
- Test Accuracy: 77.19% | Macro F1: 0.7719
- Train: 812,130 | Val: 96,518 | Test: 96,519
- Epochs: 3 | LR: 2e-5 | BS: 16 effective | fp16: True
- Labels: {0: non-sarcastic, 1: sarcastic}
- Reuse: Feature extractor backbone for PA fine-tuning ONLY

#### xlmr_emotion_model
- Architecture: XLM-RoBERTa base
- Task: 28-class GoEmotions emotion
- id2label: 28 fine-grained emotion labels including anger, annoyance
- Reuse: Auxiliary emotion feature extractor

#### xlmr_sentiment_model
- Architecture: XLM-RoBERTa base
- Task: 3-class sentiment Negative/Neutral/Positive
- Reuse: Auxiliary sentiment signal for polarity contrast detection

---

## 4. Architecture Plan for New Project

### Project Root
d:\python_folder\ (frontend preserved in place)
d:\python_folder\backend\ (new FastAPI PA backend)
d:\python_folder\training\ (training scripts)
d:\python_folder\data\ (datasets)
d:\python_folder\models\ (trained PA models)
d:\python_folder\reports\ (audit + metrics)

### Classification Architecture
- Stage A: neutral vs non-neutral (binary)
- Stage B: passive_aggressive vs overt_hostile among non-neutral
- Alternative: Direct 3-class (compare both)
- Tactic: Multi-label 6-class detection
- Models: Majority class -> LogReg -> SVM -> LightGBM -> XLM-R fine-tuned

### Reuse Decision Table

| Asset | Class | Decision |
|-------|-------|----------|
| XLM-R sarcasm weights | C | Use as encoder backbone for PA fine-tuning |
| XLM-R emotion weights | C | Auxiliary emotion features only |
| XLM-R sentiment weights | C | Auxiliary polarity features only |
| TEXT_SARCASM texts | B | Hard-negative source — sarcasm != PA |
| GoEmotions anger/annoyance | B | Potential overt_hostile candidates |
| SENTIMENT dataset | B | Hard negatives for neutral class |
| iSarcasm | F | No text available — cannot use |
| GenZ-Sarc | B | Hard negatives synthetic sarcasm |
| NLP_PROJECT backend | E | Different task incompatible |

---

## 5. Critical Constraints

1. iSarcasm has NO TEXT — tweet_id only, cannot be used
2. SARC train CSV is malformed — needs custom parser
3. No PA-labeled dataset exists in NLP_PROJECT — must build
4. FastAPI/uvicorn/spaCy not in base Python env — installing
5. XLM-R models labeled for sarcasm/emotion — must NOT be used as PA classifiers directly

---

## 6. GATE 1 Completion Status

| Task | Status |
|------|--------|
| Environment audited | COMPLETE |
| Frontend audited d:\python_folder | COMPLETE |
| NLP_PROJECT dataset inventory | COMPLETE |
| NLP_PROJECT model inventory | COMPLETE |
| Project directories created | COMPLETE |
| PDF attachment | NOT ACCESSIBLE |
| Packages installing | IN PROGRESS |

GATE 1 COMPLETE. Proceeding to GATE 2.

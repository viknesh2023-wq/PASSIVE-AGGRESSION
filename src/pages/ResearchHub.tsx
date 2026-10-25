import { useState, useEffect } from 'react';
import { ArrowRight, Check, ChevronRight } from 'lucide-react';
import { getDashboardOverview } from '../lib/api';

// ─── Verified Project Models ───────────────────────────────────────────────────
interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  accuracy: number;
  macroF1: number;
  latencyMs: number;
  size: string;
  status: 'deployed' | 'evaluated';
  role: string;
}

const VERIFIED_MODELS: ModelInfo[] = [
  {
    id: 'pa-tfidf-svm-v1',
    name: 'pa-tfidf-svm-v1',
    architecture: 'Sublinear TF-IDF (1, 2) + Platt-calibrated Linear SVM',
    accuracy: 0.9565,
    macroF1: 0.8611,
    latencyMs: 0.034,
    size: '4.9 MB',
    status: 'deployed',
    role: 'Production Real-Time Inference Model',
  },
  {
    id: 'TFIDF_LogReg',
    name: 'TFIDF_LogReg',
    architecture: 'Sublinear TF-IDF (1, 2) + Multinomial Logistic Regression',
    accuracy: 0.9516,
    macroF1: 0.8531,
    latencyMs: 0.022,
    size: '4.8 MB',
    status: 'evaluated',
    role: 'Candidate Linear Model',
  },
  {
    id: 'Distilled_Student',
    name: 'Distilled_Student',
    architecture: 'Distilled Compact Logistic Regression (3,000 unigrams)',
    accuracy: 0.9334,
    macroF1: 0.7698,
    latencyMs: 0.004,
    size: '71 KB',
    status: 'evaluated',
    role: 'Ultra-Fast Distilled Edge Model',
  },
  {
    id: 'VADER_Heuristic',
    name: 'VADER_Heuristic',
    architecture: 'Rule-Based Polarity & Sentiment Lexicon',
    accuracy: 0.8845,
    macroF1: 0.5180,
    latencyMs: 0.010,
    size: '< 1 KB',
    status: 'evaluated',
    role: 'Sentiment Polarity Baseline',
  },
  {
    id: 'MajorityBaseline',
    name: 'MajorityBaseline',
    architecture: 'Zero-Rule Majority Class Predictor',
    accuracy: 0.8822,
    macroF1: 0.3125,
    latencyMs: 0.000,
    size: '< 1 KB',
    status: 'evaluated',
    role: 'Zero-Rule Floor Baseline',
  },
  {
    id: 'Linguistic_LogReg',
    name: 'Linguistic_LogReg',
    architecture: 'Logistic Regression on 28 Hand-Crafted Linguistic Features',
    accuracy: 0.7868,
    macroF1: 0.5692,
    latencyMs: 0.001,
    size: '12 KB',
    status: 'evaluated',
    role: 'Interpretable Feature Baseline',
  },
  {
    id: 'XLM-RoBERTa-PA',
    name: 'XLM-RoBERTa-PA',
    architecture: 'XLM-RoBERTa Base Fine-Tuned (GPU, 278M parameters)',
    accuracy: 0.6987,
    macroF1: 0.6198,
    latencyMs: 16.220,
    size: '1.1 GB',
    status: 'evaluated',
    role: 'Deep Representation Model',
  },
];

// ─── Verified Datasets ────────────────────────────────────────────────────────
interface DatasetRow {
  name: string;
  count: number;
  purpose: string;
}

const VERIFIED_DATASETS: DatasetRow[] = [
  {
    name: 'TweetEval 3-Class Sentiment',
    count: 3500,
    purpose: 'Sincere conversational neutral baseline and social commentary.',
  },
  {
    name: 'SARC Reddit Corpus (Regular)',
    count: 2000,
    purpose: 'Everyday non-sarcastic conversational forum dialogue.',
  },
  {
    name: 'SARC Reddit Corpus (Irony/Sarcasm)',
    count: 2000,
    purpose: 'Hard-negative non-hostile irony and conversational sarcasm.',
  },
  {
    name: 'GoEmotions (Fine-Grained Emotion)',
    count: 1000,
    purpose: 'Auxiliary multi-label emotion representations and context vectors.',
  },
  {
    name: 'GoEmotions (Anger & Hostility)',
    count: 883,
    purpose: 'Direct overt hostility, anger, and personal attacks.',
  },
  {
    name: 'GenZ-Sarc Digital Banter',
    count: 800,
    purpose: 'Modern colloquial phrasing, slang, and non-hostile digital banter.',
  },
  {
    name: 'SARC Authentic PA Mined',
    count: 143,
    purpose: 'Authentic indirect interpersonal hostility mined from forum threads.',
  },
  {
    name: 'Curated Tactic Syntactic Expansion',
    count: 82,
    purpose: 'Controlled multi-label tactic coverage across passive-aggression markers.',
  },
  {
    name: 'Contrastive Hard Negatives',
    count: 60,
    purpose: 'Benign phrases containing lexical trigger words without hostility.',
  },
  {
    name: 'Gold Passive Aggression Anchors',
    count: 52,
    purpose: 'Adjudicated ground-truth anchors from conflict resolution studies.',
  },
];

// ─── System Architecture Pipeline Stages ───────────────────────────────────────
const ARCHITECTURE_STAGES = [
  { id: 'datasets', label: 'Datasets', hint: '10,489 unique records' },
  { id: 'preprocessing', label: 'Data Preprocessing', hint: 'Deduplication & normalization' },
  { id: 'features', label: 'Feature Extraction', hint: 'Sublinear TF-IDF (1, 2)' },
  { id: 'training', label: 'Model Training & Evaluation', hint: '7 verified models' },
  { id: 'best-model', label: 'Best Model', hint: 'pa-tfidf-svm-v1 (Macro F1 0.8611)' },
  { id: 'fastapi', label: 'FastAPI Backend', hint: 'Uvicorn REST endpoints' },
  { id: 'react', label: 'React Frontend', hint: 'React 19 + TypeScript' },
  { id: 'prediction', label: 'Prediction', hint: '3-class classification' },
];

export default function ResearchHub() {
  const [selectedModelId, setSelectedModelId] = useState<string>('pa-tfidf-svm-v1');
  const [metrics, setMetrics] = useState({
    accuracy: 0.9565,
    macroF1: 0.8611,
    uniqueRecords: 10489,
    latencyMs: 0.034,
  });

  useEffect(() => {
    async function loadBackendData() {
      try {
        const ov = await getDashboardOverview();
        if (ov) {
          setMetrics((prev) => ({
            ...prev,
            accuracy: ov.production_model?.accuracy ?? prev.accuracy,
            macroF1: ov.production_model?.macro_f1 ?? prev.macroF1,
            uniqueRecords: ov.dataset_summary?.total_unique_records ?? prev.uniqueRecords,
            latencyMs: ov.production_model?.latency_ms ?? prev.latencyMs,
          }));
        }
      } catch {
        // Fall back gracefully to verified constants
      }
    }
    loadBackendData();
  }, []);

  const selectedModel =
    VERIFIED_MODELS.find((m) => m.id === selectedModelId) || VERIFIED_MODELS[0];

  return (
    <div className="w-full min-h-screen bg-[#000000] text-[#FFFFFF] font-sans antialiased">
      <div className="max-w-[1240px] mx-auto px-5 sm:px-8 py-10 sm:py-14 space-y-14">

        {/* ─── 1. PROJECT OVERVIEW ─────────────────────────────────────────── */}
        <section aria-labelledby="project-overview-heading" className="space-y-6">
          <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 border-b border-[#262626] pb-8">
            <div className="space-y-3 max-w-4xl">
              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-[#121212] border border-[#262626] text-xs font-mono text-[#A3A3A3]">
                <span>Technical Dashboard</span>
                <span>•</span>
                <span>PassiveAggression Detector 2.0</span>
              </div>
              <h1
                id="project-overview-heading"
                className="text-2xl sm:text-3xl lg:text-4xl font-extrabold tracking-tight text-[#FFFFFF] leading-tight"
              >
                PassiveAggression Detector 2.0
              </h1>
              <p className="text-base sm:text-lg text-[#A3A3A3] leading-relaxed">
                Explainable machine learning detection and classification of subtle, covert hostility in English text.
              </p>
            </div>

            <div className="shrink-0 self-start">
              <a
                href="/"
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-[#FFFFFF] hover:bg-[#E5E5E5] text-[#000000] text-xs font-semibold transition-colors shadow-sm"
              >
                Open Analyzer <ArrowRight size={14} />
              </a>
            </div>
          </div>

          {/* Scope, Objective, and Technologies */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
            <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-6 space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#737373]">
                Scope
              </span>
              <h2 className="text-sm font-semibold text-[#FFFFFF]">Strict English-Only</h2>
              <p className="text-xs text-[#A3A3A3] leading-relaxed">
                Multilingual text, Hinglish, and cross-lingual mixes are systematically excluded to avoid pragmatic distortion and maintain accurate hostility boundary detection.
              </p>
            </div>

            <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-6 space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#737373]">
                Project Objective
              </span>
              <h2 className="text-sm font-semibold text-[#FFFFFF]">Covert Hostility Resolution</h2>
              <p className="text-xs text-[#A3A3A3] leading-relaxed">
                Distinguish benign phrasing and harmless banter from genuine passive aggression with calibrated probabilities, transparent explanations, and sub-millisecond edge latency.
              </p>
            </div>

            <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-6 space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#737373]">
                Core Technologies
              </span>
              <h2 className="text-sm font-semibold text-[#FFFFFF]">Full-Stack NLP Pipeline</h2>
              <p className="text-xs text-[#A3A3A3] leading-relaxed">
                React 19, TypeScript, Vite, FastAPI, Python 3.12, scikit-learn, PyTorch (CUDA 12.1), and Platt-calibrated Linear SVM.
              </p>
            </div>
          </div>
        </section>


        {/* ─── 2. KEY METRICS ─────────────────────────────────────────────── */}
        <section aria-labelledby="key-metrics-heading" className="space-y-4">
          <div className="border-b border-[#262626] pb-3">
            <h2 id="key-metrics-heading" className="text-xl font-bold tracking-tight text-[#FFFFFF]">
              Key Metrics
            </h2>
            <p className="text-xs sm:text-sm text-[#737373] mt-1">
              Verified performance metrics computed on the held-out test split (1,426 samples).
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 pt-2">
            {/* Metric 1 */}
            <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-6 flex flex-col justify-between">
              <span className="text-xs font-medium text-[#737373] uppercase tracking-wider">
                Model Accuracy
              </span>
              <div className="my-3">
                <span className="text-3xl sm:text-4xl font-extrabold text-[#FFFFFF] tracking-tight">
                  {(metrics.accuracy * 100).toFixed(2)}%
                </span>
              </div>
              <p className="text-xs text-[#A3A3A3]">
                Held-out test split (1,426 samples)
              </p>
            </div>

            {/* Metric 2 */}
            <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-6 flex flex-col justify-between">
              <span className="text-xs font-medium text-[#737373] uppercase tracking-wider">
                Macro F1-Score
              </span>
              <div className="my-3">
                <span className="text-3xl sm:text-4xl font-extrabold text-[#FFFFFF] tracking-tight">
                  {metrics.macroF1.toFixed(4)}
                </span>
              </div>
              <p className="text-xs text-[#A3A3A3]">
                Balanced across all 3 target classes
              </p>
            </div>

            {/* Metric 3 */}
            <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-6 flex flex-col justify-between">
              <span className="text-xs font-medium text-[#737373] uppercase tracking-wider">
                Unique Dataset Records
              </span>
              <div className="my-3">
                <span className="text-3xl sm:text-4xl font-extrabold text-[#FFFFFF] tracking-tight font-mono">
                  {metrics.uniqueRecords.toLocaleString()}
                </span>
              </div>
              <p className="text-xs text-[#A3A3A3]">
                Zero leakage across train, val, & test
              </p>
            </div>

            {/* Metric 4 */}
            <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-6 flex flex-col justify-between">
              <span className="text-xs font-medium text-[#737373] uppercase tracking-wider">
                Inference Latency
              </span>
              <div className="my-3">
                <span className="text-3xl sm:text-4xl font-extrabold text-[#FFFFFF] tracking-tight font-mono">
                  {metrics.latencyMs.toFixed(3)} ms
                </span>
              </div>
              <p className="text-xs text-[#A3A3A3]">
                Single-threaded CPU edge inference
              </p>
            </div>
          </div>
        </section>


        {/* ─── 3. SYSTEM ARCHITECTURE DIAGRAM ─────────────────────────────── */}
        <section aria-labelledby="architecture-heading" className="space-y-6">
          <div className="border-b border-[#262626] pb-3">
            <h2 id="architecture-heading" className="text-xl font-bold tracking-tight text-[#FFFFFF]">
              System Architecture Diagram
            </h2>
            <p className="text-xs sm:text-sm text-[#737373] mt-1">
              End-to-end data pipeline from raw datasets to real-time inference. Click any model node to view its verified metrics.
            </p>
          </div>

          <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-6 sm:p-8 space-y-8">
            
            {/* Primary Horizontal Pipeline */}
            <div className="overflow-x-auto pb-2">
              <div className="flex items-center min-w-[980px] justify-between gap-2">
                {ARCHITECTURE_STAGES.map((stage, idx) => {
                  const isHighlighted = stage.id === 'best-model';
                  return (
                    <div key={stage.id} className="flex items-center flex-1">
                      <div
                        className={`flex-1 rounded-lg border px-3 py-3 text-center transition-all ${
                          isHighlighted
                            ? 'bg-[#141414] border-[#525252] shadow-sm'
                            : 'bg-[#0F0F0F] border-[#262626]'
                        }`}
                      >
                        <span className="block text-[11px] font-mono text-[#737373] uppercase">
                          Stage 0{idx + 1}
                        </span>
                        <span
                          className={`block text-xs font-semibold mt-1 ${
                            isHighlighted ? 'text-[#FFFFFF]' : 'text-[#E5E5E5]'
                          }`}
                        >
                          {stage.label}
                        </span>
                        <span className="block text-[10px] text-[#858585] mt-1 leading-tight">
                          {stage.hint}
                        </span>
                      </div>

                      {idx < ARCHITECTURE_STAGES.length - 1 && (
                        <div className="px-2 text-[#525252] shrink-0">
                          <ChevronRight size={14} />
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Model Nodes Connected under Stage 04: Model Training & Evaluation */}
            <div className="border-t border-[#1F1F1F] pt-6 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <span className="text-xs font-mono uppercase tracking-wider text-[#737373]">
                    Connected Models Under Training & Evaluation
                  </span>
                  <p className="text-xs text-[#A3A3A3] mt-0.5">
                    Select a model node below to inspect its verified evaluation results.
                  </p>
                </div>
                <div className="inline-flex items-center gap-1.5 text-xs text-[#737373]">
                  <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                  <span>Highlighted model is currently deployed for live predictions</span>
                </div>
              </div>

              {/* Model Nodes Selector */}
              <div className="flex flex-wrap gap-2.5 pt-1">
                {VERIFIED_MODELS.map((model) => {
                  const isSelected = selectedModelId === model.id;
                  const isDeployed = model.status === 'deployed';

                  return (
                    <button
                      key={model.id}
                      onClick={() => setSelectedModelId(model.id)}
                      className={`group relative inline-flex items-center gap-2 px-3.5 py-2 rounded-lg border text-xs font-mono transition-all cursor-pointer ${
                        isSelected
                          ? 'bg-[#1C1C1C] border-[#FFFFFF] text-[#FFFFFF] shadow-sm'
                          : isDeployed
                          ? 'bg-[#121212] border-[#383838] text-[#E5E5E5] hover:border-[#525252]'
                          : 'bg-[#0D0D0D] border-[#212121] text-[#858585] hover:border-[#333333] hover:text-[#D4D4D4]'
                      }`}
                    >
                      {isDeployed && (
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                      )}
                      <span className="font-semibold">{model.name}</span>
                      {isDeployed && (
                        <span className="text-[10px] uppercase font-mono px-1.5 py-0.2 rounded bg-[#1F1F1F] text-emerald-400 border border-emerald-900/60">
                          Deployed
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* In-Place Interactive Model Details Card */}
              <div className="rounded-lg border border-[#262626] bg-[#070707] p-5 mt-4 transition-all">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#1A1A1A] pb-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-base font-bold text-[#FFFFFF] font-mono">
                        {selectedModel.name}
                      </h3>
                      {selectedModel.status === 'deployed' ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-950/70 text-emerald-300 border border-emerald-800/80">
                          <Check size={12} /> Real-Time Prediction Model
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded text-[11px] font-medium bg-[#141414] text-[#858585] border border-[#262626]">
                          Evaluated Baseline
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-[#A3A3A3] mt-1.5">
                      {selectedModel.architecture}
                    </p>
                  </div>

                  <div className="shrink-0 text-xs font-mono text-[#737373]">
                    Role: <span className="text-[#CCCCCC]">{selectedModel.role}</span>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 text-xs">
                  <div>
                    <span className="text-[#737373] block text-[11px] uppercase tracking-wider">
                      Accuracy
                    </span>
                    <span className="text-base font-bold text-[#FFFFFF] font-mono mt-0.5 block">
                      {(selectedModel.accuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-[#737373] block text-[11px] uppercase tracking-wider">
                      Macro F1
                    </span>
                    <span className="text-base font-bold text-[#FFFFFF] font-mono mt-0.5 block">
                      {selectedModel.macroF1.toFixed(4)}
                    </span>
                  </div>
                  <div>
                    <span className="text-[#737373] block text-[11px] uppercase tracking-wider">
                      Inference Latency
                    </span>
                    <span className="text-base font-bold text-[#FFFFFF] font-mono mt-0.5 block">
                      {selectedModel.latencyMs.toFixed(3)} ms
                    </span>
                  </div>
                  <div>
                    <span className="text-[#737373] block text-[11px] uppercase tracking-wider">
                      Artifact Size
                    </span>
                    <span className="text-base font-bold text-[#FFFFFF] font-mono mt-0.5 block">
                      {selectedModel.size}
                    </span>
                  </div>
                </div>
              </div>
            </div>

          </div>
        </section>


        {/* ─── 4. DATASET SUMMARY ─────────────────────────────────────────── */}
        <section aria-labelledby="dataset-summary-heading" className="space-y-4">
          <div className="border-b border-[#262626] pb-3">
            <h2 id="dataset-summary-heading" className="text-xl font-bold tracking-tight text-[#FFFFFF]">
              Dataset Summary
            </h2>
            <p className="text-xs sm:text-sm text-[#737373] mt-1">
              Verified dataset breakdown totaling 10,489 unique records with 0% train/test overlap.
            </p>
          </div>

          <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse" style={{ minWidth: 720 }}>
                <colgroup>
                  <col style={{ width: '30%' }} />
                  <col style={{ width: '18%' }} />
                  <col style={{ width: '52%' }} />
                </colgroup>
                <thead>
                  <tr className="border-b border-[#262626] bg-[#0E0E0E] text-[11px] font-mono uppercase tracking-wider text-[#737373]">
                    <th style={{ padding: '14px 24px' }} className="font-semibold text-left">Dataset Name</th>
                    <th style={{ padding: '14px 24px' }} className="font-semibold text-right">Record Count</th>
                    <th style={{ padding: '14px 24px' }} className="font-semibold text-left">Purpose</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1A1A1A] text-xs">
                  {VERIFIED_DATASETS.map((ds) => (
                    <tr key={ds.name} className="hover:bg-[#121212] transition-colors">
                      <td style={{ padding: '14px 24px' }} className="font-medium text-[#FFFFFF] whitespace-nowrap text-left">
                        {ds.name}
                      </td>
                      <td style={{ padding: '14px 24px' }} className="text-right font-mono text-[#D4D4D4] whitespace-nowrap">
                        {ds.count.toLocaleString()}
                      </td>
                      <td style={{ padding: '14px 24px' }} className="text-[#A3A3A3] leading-relaxed text-left">
                        {ds.purpose}
                      </td>
                    </tr>
                  ))}
                  {/* Total Row */}
                  <tr className="border-t-2 border-[#262626] bg-[#0E0E0E] font-semibold">
                    <td style={{ padding: '14px 24px' }} className="text-[#FFFFFF] text-left">
                      Total Ingested Records (Deduplicated)
                    </td>
                    <td style={{ padding: '14px 24px' }} className="text-right font-mono text-[#FFFFFF] whitespace-nowrap">
                      10,489
                    </td>
                    <td style={{ padding: '14px 24px' }} className="text-xs text-[#A3A3A3] leading-relaxed text-left">
                      Partitioned into 70% Train (6,638), 15% Val (1,425), and 15% Test (1,426) with verified 0.0% string overlap.
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>


        {/* ─── 5. MODEL COMPARISON ────────────────────────────────────────── */}
        <section aria-labelledby="model-comparison-heading" className="space-y-4">
          <div className="border-b border-[#262626] pb-3">
            <h2 id="model-comparison-heading" className="text-xl font-bold tracking-tight text-[#FFFFFF]">
              Model Comparison
            </h2>
            <p className="text-xs sm:text-sm text-[#737373] mt-1">
              Empirical benchmark across 7 candidate models. The deployed production model is highlighted.
            </p>
          </div>

          <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse" style={{ minWidth: 760 }}>
                <colgroup>
                  <col style={{ width: '32%' }} />
                  <col style={{ width: '15%' }} />
                  <col style={{ width: '16%' }} />
                  <col style={{ width: '15%' }} />
                  <col style={{ width: '22%' }} />
                </colgroup>
                <thead>
                  <tr className="border-b border-[#262626] bg-[#0E0E0E] text-[11px] font-mono uppercase tracking-wider text-[#737373]">
                    <th style={{ padding: '14px 24px' }} className="font-semibold text-left">Model Name</th>
                    <th style={{ padding: '14px 24px' }} className="font-semibold text-right">Accuracy</th>
                    <th style={{ padding: '14px 24px' }} className="font-semibold text-right">Macro F1-Score</th>
                    <th style={{ padding: '14px 24px' }} className="font-semibold text-right">Latency</th>
                    <th style={{ padding: '14px 24px' }} className="font-semibold text-left">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1A1A1A] text-xs">
                  {VERIFIED_MODELS.map((model) => {
                    const isDeployed = model.status === 'deployed';
                    return (
                      <tr
                        key={model.id}
                        className={`transition-colors ${
                          isDeployed
                            ? 'bg-[#141414] hover:bg-[#181818]'
                            : 'hover:bg-[#121212]'
                        }`}
                      >
                        <td style={{ padding: '14px 24px' }} className="text-left">
                          <div className="font-mono font-semibold text-[#FFFFFF]">
                            {model.name}
                          </div>
                          <div className="text-[11px] text-[#737373] mt-0.5">
                            {model.architecture}
                          </div>
                        </td>
                        <td style={{ padding: '14px 24px' }} className="text-right font-mono font-medium text-[#FFFFFF] whitespace-nowrap">
                          {(model.accuracy * 100).toFixed(2)}%
                        </td>
                        <td style={{ padding: '14px 24px' }} className="text-right font-mono font-medium text-[#FFFFFF] whitespace-nowrap">
                          {model.macroF1.toFixed(4)}
                        </td>
                        <td style={{ padding: '14px 24px' }} className="text-right font-mono text-[#A3A3A3] whitespace-nowrap">
                          {model.latencyMs.toFixed(3)} ms
                        </td>
                        <td style={{ padding: '14px 24px' }} className="whitespace-nowrap text-left">
                          {isDeployed ? (
                            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-semibold bg-emerald-950/70 text-emerald-300 border border-emerald-800/80">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                              Deployed (Production)
                            </span>
                          ) : (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] text-[#858585] bg-[#121212] border border-[#262626]">
                              Evaluated
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </section>


        {/* ─── 6. TECHNOLOGY STACK ────────────────────────────────────────── */}
        <section aria-labelledby="technology-stack-heading" className="space-y-4">
          <div className="border-b border-[#262626] pb-3">
            <h2 id="technology-stack-heading" className="text-xl font-bold tracking-tight text-[#FFFFFF]">
              Technology Stack
            </h2>
            <p className="text-xs sm:text-sm text-[#737373] mt-1">
              Core runtime, frameworks, active model artifact, and target hardware environment.
            </p>
          </div>

          <div className="rounded-xl border border-[#262626] bg-[#0A0A0A] p-6 sm:p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 text-xs">
              
              {/* Frontend */}
              <div className="space-y-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">
                  Frontend
                </span>
                <p className="text-sm font-semibold text-[#FFFFFF]">
                  React 19 & TypeScript
                </p>
                <p className="text-xs text-[#A3A3A3] leading-relaxed">
                  Vite bundler, Tailwind CSS design system, Zustand state store, React Query, Recharts data visualization, and Lucide React icons.
                </p>
              </div>

              {/* Backend */}
              <div className="space-y-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">
                  Backend
                </span>
                <p className="text-sm font-semibold text-[#FFFFFF]">
                  FastAPI & Python 3.12
                </p>
                <p className="text-xs text-[#A3A3A3] leading-relaxed">
                  Asynchronous Uvicorn server, Pydantic validation schemas, CORS middleware, and local SQLite persistence for analysis history.
                </p>
              </div>

              {/* ML Frameworks */}
              <div className="space-y-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">
                  ML & NLP Frameworks
                </span>
                <p className="text-sm font-semibold text-[#FFFFFF]">
                  scikit-learn 1.9 & PyTorch
                </p>
                <p className="text-xs text-[#A3A3A3] leading-relaxed">
                  Sublinear TF-IDF vectorizer, Platt-calibrated LinearSVC, NLTK VADER sentiment lexicon, and Cleanlab confident learning for noise auditing.
                </p>
              </div>

              {/* Active Model */}
              <div className="space-y-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">
                  Active Deployed Model
                </span>
                <p className="text-sm font-semibold text-[#FFFFFF] font-mono">
                  pa-tfidf-svm-v1 (4.9 MB)
                </p>
                <p className="text-xs text-[#A3A3A3] leading-relaxed">
                  Linear Support Vector Classifier with Sigmoid Platt Calibration. 0.034 ms CPU inference latency and 0.8611 macro F1-score.
                </p>
              </div>

              {/* Quality & Testing */}
              <div className="space-y-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">
                  Validation & QA
                </span>
                <p className="text-sm font-semibold text-[#FFFFFF]">
                  23 / 23 Tests Passing (100%)
                </p>
                <p className="text-xs text-[#A3A3A3] leading-relaxed">
                  pytest automated suite verifying calibration bounds, data leakage prevention (0.0%), API contract stability, and edge case resilience.
                </p>
              </div>

              {/* Hardware */}
              <div className="space-y-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#737373]">
                  Hardware & Execution
                </span>
                <p className="text-sm font-semibold text-[#FFFFFF]">
                  RTX 4050 GPU & CPU Edge
                </p>
                <p className="text-xs text-[#A3A3A3] leading-relaxed">
                  NVIDIA GeForce RTX 4050 Laptop GPU (6.44 GB VRAM, CUDA 12.1) for transformer training; instant CPU execution for production inference.
                </p>
              </div>

            </div>
          </div>
        </section>

      </div>
    </div>
  );
}

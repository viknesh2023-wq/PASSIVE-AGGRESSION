import { useState, useRef, useMemo } from 'react';
import {
  ZoomIn, ZoomOut, Maximize2, RotateCcw, Filter,
  Layers, Sliders, Cpu, ArrowRight,
  Zap, Lock
} from 'lucide-react';

export interface ArchNode {
  id: string;
  name: string;
  layer: 'dataset' | 'preprocess' | 'features' | 'model' | 'eval' | 'api' | 'frontend';
  col: number; // 0 to 6
  row: number; // order in column
  status: 'production' | 'evaluated' | 'verified' | 'baseline' | 'experimental' | 'isolated';
  description: string;
  file?: string;
  artifact?: string;
  metrics?: { label: string; value: string }[];
  inputs: string[];
  outputs: string[];
  details?: string[];
  isPrimaryInference?: boolean;
}

export interface ArchEdge {
  from: string;
  to: string;
  label?: string;
  type?: 'train' | 'eval' | 'inference' | 'data' | 'leakage_guard';
}

const ARCH_NODES: ArchNode[] = [
  // ── Column 0: Dataset Sources ──
  {
    id: 'src_sarc',
    name: 'SARC Reddit Sarcasm',
    layer: 'dataset',
    col: 0,
    row: 0,
    status: 'verified',
    description: '2,500 conversational dialogue turns extracted from the Stanford SARC corpus.',
    file: 'training/build_large_scale_corpus.py',
    artifact: 'data/raw/sarc_reddit.csv',
    metrics: [{ label: 'Records', value: '2,500' }, { label: 'Class', value: 'PA & Neutral' }],
    inputs: ['Open-Source Reddit Corpus'],
    outputs: ['Corpus Stream (English)']
  },
  {
    id: 'src_tweeteval',
    name: 'TweetEval Irony & Stance',
    layer: 'dataset',
    col: 0,
    row: 1,
    status: 'verified',
    description: '2,000 short-form English social turns with verified sentiment annotations.',
    file: 'training/build_large_scale_corpus.py',
    artifact: 'data/raw/tweeteval_irony.csv',
    metrics: [{ label: 'Records', value: '2,000' }, { label: 'Class', value: 'Hostile / Irony' }],
    inputs: ['CardiffNLP TweetEval'],
    outputs: ['Corpus Stream (English)']
  },
  {
    id: 'src_genz',
    name: 'Gen-Z Workplace Corpus',
    layer: 'dataset',
    col: 0,
    row: 2,
    status: 'verified',
    description: '1,800 workplace Slack, email, and peer dialogue snippets with passive-aggressive phrasing.',
    file: 'training/build_large_scale_corpus.py',
    artifact: 'data/raw/genz_corpus.csv',
    metrics: [{ label: 'Records', value: '1,800' }, { label: 'Class', value: 'Workplace PA' }],
    inputs: ['Curated Workplace Transcripts'],
    outputs: ['Corpus Stream (English)']
  },
  {
    id: 'src_goemotions',
    name: 'GoEmotions Anger & Annoyance',
    layer: 'dataset',
    col: 0,
    row: 3,
    status: 'verified',
    description: '1,500 Reddit comments filtered for subtle irritation, annoyance, and covert disapproval.',
    file: 'training/build_large_scale_corpus.py',
    artifact: 'data/raw/goemotions.csv',
    metrics: [{ label: 'Records', value: '1,500' }, { label: 'Class', value: 'Covert Hostility' }],
    inputs: ['Google Research GoEmotions'],
    outputs: ['Corpus Stream (English)']
  },
  {
    id: 'src_pa_seed',
    name: 'Passive-Aggression Seed Corpus',
    layer: 'dataset',
    col: 0,
    row: 4,
    status: 'verified',
    description: '2,689 gold-annotated passive-aggressive communications from workplace conflict studies.',
    file: 'data/processed/pa_dataset.csv',
    artifact: 'data/processed/pa_dataset.csv',
    metrics: [{ label: 'Records', value: '2,689' }, { label: 'Total Unique', value: '10,489' }],
    inputs: ['Conflict Resolution Datasets'],
    outputs: ['Multi-Source Corpus (10,489)']
  },

  // ── Column 1: Ingestion & Preprocessing ──
  {
    id: 'prep_schema',
    name: 'Schema Normalization & NFC',
    layer: 'preprocess',
    col: 1,
    row: 0,
    status: 'verified',
    description: 'Normalizes Unicode to NFC, unifies Windows CRLF line endings, trims whitespace.',
    file: 'backend/app/nlp/preprocess.py',
    inputs: ['All 10,489 Raw Records'],
    outputs: ['Clean Text Strings']
  },
  {
    id: 'prep_lang',
    name: 'English-Only Script Filter',
    layer: 'preprocess',
    col: 1,
    row: 1,
    status: 'verified',
    isPrimaryInference: true,
    description: 'Validates English Latin script while strictly allowing valid workplace slang, abbreviations, and emojis.',
    file: 'backend/app/nlp/validate_english.py',
    inputs: ['Clean Text Strings'],
    outputs: ['Validated English Corpus']
  },
  {
    id: 'prep_dedup',
    name: 'MD5 Exact & Near Deduplication',
    layer: 'preprocess',
    col: 1,
    row: 2,
    status: 'verified',
    description: 'Eliminates cross-dataset near-duplicate strings to prevent artificial frequency inflation.',
    file: 'training/build_large_scale_corpus.py',
    inputs: ['Validated English Corpus'],
    outputs: ['Deduplicated Corpus (10,489 records)']
  },
  {
    id: 'split_train',
    name: 'Train Split (70%)',
    layer: 'preprocess',
    col: 1,
    row: 3,
    status: 'verified',
    description: '6,638 stratified training examples. Strictly quarantined from evaluation.',
    file: 'reports/split_integrity_report.md',
    metrics: [{ label: 'Count', value: '6,638' }, { label: 'Leakage', value: '0%' }],
    inputs: ['Stratified Splitter'],
    outputs: ['Model Training Pipeline']
  },
  {
    id: 'split_val',
    name: 'Validation Split (15%)',
    layer: 'preprocess',
    col: 1,
    row: 4,
    status: 'verified',
    description: '1,425 records used strictly for threshold tuning, Platt calibration, and early stopping.',
    file: 'reports/split_integrity_report.md',
    metrics: [{ label: 'Count', value: '1,425' }, { label: 'Overlap', value: '0.0%' }],
    inputs: ['Stratified Splitter'],
    outputs: ['Platt Calibration & Tuning']
  },
  {
    id: 'split_test',
    name: 'Held-Out Test Split (15%)',
    layer: 'preprocess',
    col: 1,
    row: 5,
    status: 'isolated',
    description: '1,426 isolated test examples. ZERO training connection! Never seen during model fitting.',
    file: 'reports/split_integrity_report.md',
    metrics: [{ label: 'Count', value: '1,426' }, { label: 'Status', value: 'Strictly Isolated' }],
    inputs: ['Stratified Splitter'],
    outputs: ['Held-Out Test Benchmark']
  },

  // ── Column 2: Feature Engineering & Quality ──
  {
    id: 'feat_tfidf',
    name: 'Sublinear TF-IDF (1, 2)',
    layer: 'features',
    col: 2,
    row: 0,
    status: 'production',
    isPrimaryInference: true,
    description: 'Word & char n-grams (1, 2) with sublinear term-frequency scaling and min_df=2.',
    file: 'models/pa_vectorizer.pkl',
    metrics: [{ label: 'Vocabulary', value: '14,820' }, { label: 'Type', value: 'Sparse Float32' }],
    inputs: ['Train Split (6,638)'],
    outputs: ['Sparse Feature Vectors']
  },
  {
    id: 'feat_incongruity',
    name: 'Inter-Clause Incongruity',
    layer: 'features',
    col: 2,
    row: 1,
    status: 'verified',
    description: 'Computes compound sentiment divergence between greeting clause and body critique.',
    file: 'training/incongruity_model.py',
    metrics: [{ label: 'Features', value: '8 dims' }, { label: 'F1 Gain', value: '+1.8%' }],
    inputs: ['Tokenized Sentences'],
    outputs: ['Incongruity Vector']
  },
  {
    id: 'feat_phrases',
    name: 'Curated 6-Tactic Matcher',
    layer: 'features',
    col: 2,
    row: 2,
    status: 'production',
    isPrimaryInference: true,
    description: 'Exact character-offset matcher tracking sarcasm, blame-shifting, mock politeness, and curt dismissals.',
    file: 'backend/app/nlp/phrase_matcher.py',
    metrics: [{ label: 'Tactics', value: '6 classes' }, { label: 'Resolution', value: 'Char offsets' }],
    inputs: ['Raw Message Text'],
    outputs: ['Highlight Spans & Reasons']
  },
  {
    id: 'feat_cleanlab',
    name: 'Cleanlab Noise Audit',
    layer: 'features',
    col: 2,
    row: 3,
    status: 'verified',
    description: '5-fold out-of-sample confident learning audit identifying 118 label noise candidates.',
    file: 'training/label_noise_audit.py',
    metrics: [{ label: 'Issues Found', value: '118' }, { label: 'Filtered F1', value: '0.8690' }],
    inputs: ['Cross-Validation Probabilities'],
    outputs: ['Cleaned Training Labels']
  },

  // ── Column 3: Model Architectures ──
  {
    id: 'model_svm',
    name: 'pa-tfidf-svm-v1 (Linear SVM)',
    layer: 'model',
    col: 3,
    row: 0,
    status: 'production',
    isPrimaryInference: true,
    description: 'Production Deployed Model: Calibrated Linear SVM with Platt sigmoid scaling. 95.65% Acc at 0.034 ms latency.',
    file: 'models/pa_classifier.pkl',
    artifact: 'models/pa_classifier.pkl',
    metrics: [
      { label: 'Accuracy', value: '95.65%' },
      { label: 'Macro F1', value: '0.8611' },
      { label: 'PA F1', value: '0.7900' },
      { label: 'Latency', value: '0.034 ms' },
      { label: 'Size', value: '4.9 MB' }
    ],
    inputs: ['Sublinear TF-IDF Features', 'Train Split'],
    outputs: ['Calibrated 3-Class Probabilities']
  },
  {
    id: 'model_distill',
    name: 'Distilled Student Model',
    layer: 'model',
    col: 3,
    row: 1,
    status: 'evaluated',
    description: 'Compact 71 KB student model distilled from teacher probabilities for ultra-low latency edge devices.',
    file: 'training/ensemble_distillation.py',
    metrics: [
      { label: 'Accuracy', value: '93.34%' },
      { label: 'Macro F1', value: '0.7698' },
      { label: 'Latency', value: '0.004 ms' },
      { label: 'Size', value: '71 KB' }
    ],
    inputs: ['Soft Teacher Probabilities'],
    outputs: ['Edge Predictions']
  },
  {
    id: 'model_roberta',
    name: 'XLM-RoBERTa-PA Transformer',
    layer: 'model',
    col: 3,
    row: 2,
    status: 'evaluated',
    description: 'Pre-trained cross-lingual transformer fine-tuned on GPU. High expressiveness, evaluated benchmark.',
    file: 'training/train_transformer.py',
    metrics: [
      { label: 'Accuracy', value: '69.87%' },
      { label: 'Macro F1', value: '0.6198' },
      { label: 'Latency', value: '16.22 ms' },
      { label: 'Parameters', value: '278M' }
    ],
    inputs: ['Subword Token IDs', 'Train Split'],
    outputs: ['Dense Logits']
  },
  {
    id: 'model_logreg',
    name: 'Linguistic Logistic Regression',
    layer: 'model',
    col: 3,
    row: 3,
    status: 'baseline',
    description: 'Linear baseline using 28 hand-crafted linguistic markers, punctuation ratios, and sentiment scores.',
    file: 'training/train_baselines.py',
    metrics: [
      { label: 'Accuracy', value: '78.68%' },
      { label: 'Macro F1', value: '0.5692' },
      { label: 'Latency', value: '0.001 ms' }
    ],
    inputs: ['Linguistic Feature Set'],
    outputs: ['Class Probabilities']
  },
  {
    id: 'model_vader',
    name: 'VADER Sentiment Heuristic',
    layer: 'model',
    col: 3,
    row: 4,
    status: 'baseline',
    description: 'Rule-based polarity heuristic showing that negative sentiment alone does not equal passive aggression.',
    file: 'training/train_baselines.py',
    metrics: [
      { label: 'Accuracy', value: '88.45%' },
      { label: 'Macro F1', value: '0.5180' },
      { label: 'PA F1', value: '0.2400' }
    ],
    inputs: ['Raw Text'],
    outputs: ['Sentiment Polarity']
  },
  {
    id: 'model_majority',
    name: 'Majority Class Predictor',
    layer: 'model',
    col: 3,
    row: 5,
    status: 'baseline',
    description: 'Floor baseline predicting the neutral majority class across all instances (Macro F1: 0.3125).',
    file: 'training/train_baselines.py',
    metrics: [
      { label: 'Accuracy', value: '88.22%' },
      { label: 'Macro F1', value: '0.3125' },
      { label: 'PA F1', value: '0.0000' }
    ],
    inputs: ['Label Distribution'],
    outputs: ['Default Class']
  },

  // ── Column 4: Evaluation & Robustness ──
  {
    id: 'eval_heldout',
    name: 'Held-Out Test Benchmark',
    layer: 'eval',
    col: 4,
    row: 0,
    status: 'verified',
    description: 'Rigorous evaluation against 1,426 isolated test records. Computes precision, recall, and F1.',
    file: 'reports/metrics_summary.json',
    metrics: [{ label: 'Evaluated On', value: '1,426 Records' }, { label: 'Top F1', value: '0.8611' }],
    inputs: ['Held-Out Test Split (1,426)', 'Model Checkpoints'],
    outputs: ['metrics_summary.json']
  },
  {
    id: 'eval_checklist',
    name: 'CheckList Minimal-Pair Suite',
    layer: 'eval',
    col: 4,
    row: 1,
    status: 'verified',
    description: '16 adversarial perturbation tests: casing, punctuation, emoji injection, and contraction swaps.',
    file: 'training/robustness_checklist.py',
    metrics: [{ label: 'Pass Rate', value: '68.8%' }, { label: 'Tests', value: '16 pairs' }],
    inputs: ['Perturbed Variations'],
    outputs: ['Robustness Scorecard']
  },
  {
    id: 'eval_faithfulness',
    name: 'Deletion Faithfulness Tests',
    layer: 'eval',
    col: 4,
    row: 2,
    status: 'verified',
    description: 'Demonstrates an 8.01x higher confidence drop when removing detected evidence vs random words.',
    file: 'training/robustness_checklist.py',
    metrics: [{ label: 'Causal Ratio', value: '8.01x' }, { label: 'Status', value: 'Verified' }],
    inputs: ['Detected Phrase Spans'],
    outputs: ['Causal Attribution Validation']
  },

  // ── Column 5: Inference API & Runtime ──
  {
    id: 'api_fastapi',
    name: 'FastAPI REST Application',
    layer: 'api',
    col: 5,
    row: 0,
    status: 'production',
    isPrimaryInference: true,
    description: 'High-performance ASGI server on 127.0.0.1:8000 serving /api/analyze, /api/batch, /api/dashboard.',
    file: 'backend/app/main.py',
    metrics: [{ label: 'Port', value: '8000' }, { label: 'Framework', value: 'FastAPI' }],
    inputs: ['HTTP POST /api/analyze'],
    outputs: ['JSON Prediction Response']
  },
  {
    id: 'api_rewriter',
    name: 'Deterministic Calmer Rewriter',
    layer: 'api',
    col: 5,
    row: 1,
    status: 'production',
    isPrimaryInference: true,
    description: 'Curated pattern substitution mapping hostile expressions to calm, professional alternatives.',
    file: 'backend/app/nlp/rewriter.py',
    metrics: [{ label: 'Templates', value: 'Deterministic' }, { label: 'Generative LLM', value: 'None (Safe)' }],
    inputs: ['Identified Tactic Spans'],
    outputs: ['Constructive Alternative Phrasings']
  },
  {
    id: 'api_db',
    name: 'SQLite History & Feedback',
    layer: 'api',
    col: 5,
    row: 2,
    status: 'verified',
    description: 'Local relational persistence storing audit logs, user corrections, and feedback without telemetry.',
    file: 'backend/app/database',
    artifact: 'backend/pa_detector.db',
    metrics: [{ label: 'Engine', value: 'SQLite3' }, { label: 'Privacy', value: '100% Local' }],
    inputs: ['Analysis Results', 'User Feedback'],
    outputs: ['Persistent History Log']
  },

  // ── Column 6: React Frontend ──
  {
    id: 'ui_analyzer',
    name: 'Single Message Analyzer',
    layer: 'frontend',
    col: 6,
    row: 0,
    status: 'production',
    isPrimaryInference: true,
    description: 'Real-time classification UI with interactive score ring, tactic chips, and sentence breakdowns.',
    file: 'src/pages/Analyzer.tsx',
    inputs: ['User Text Input'],
    outputs: ['Visual Verdict & Highlights']
  },
  {
    id: 'ui_thread',
    name: 'Multi-Turn Thread Analyzer',
    layer: 'frontend',
    col: 6,
    row: 1,
    status: 'production',
    description: 'Conversation timeline tracking tension escalation across multiple dialogue speakers.',
    file: 'src/pages/Thread.tsx',
    inputs: ['Dialogue Transcripts'],
    outputs: ['Tension Curve & Bubble Stream']
  },
  {
    id: 'ui_dashboard',
    name: 'Technical Research Dashboard',
    layer: 'frontend',
    col: 6,
    row: 2,
    status: 'production',
    description: 'Full-stack engineering console detailing datasets, models, metrics, and architecture layers.',
    file: 'src/pages/ResearchHub.tsx',
    inputs: ['Dashboard API Endpoints'],
    outputs: ['Interactive Architecture Canvas']
  }
];

const ARCH_EDGES: ArchEdge[] = [
  // Datasets -> Preprocess
  { from: 'src_sarc', to: 'prep_schema', type: 'data' },
  { from: 'src_tweeteval', to: 'prep_schema', type: 'data' },
  { from: 'src_genz', to: 'prep_schema', type: 'data' },
  { from: 'src_goemotions', to: 'prep_schema', type: 'data' },
  { from: 'src_pa_seed', to: 'prep_schema', type: 'data' },
  { from: 'prep_schema', to: 'prep_lang', type: 'data' },
  { from: 'prep_lang', to: 'prep_dedup', type: 'data' },
  
  // Splits
  { from: 'prep_dedup', to: 'split_train', type: 'data' },
  { from: 'prep_dedup', to: 'split_val', type: 'data' },
  { from: 'prep_dedup', to: 'split_test', type: 'leakage_guard', label: 'Isolated (No Train)' },

  // Train -> Features
  { from: 'split_train', to: 'feat_tfidf', type: 'train' },
  { from: 'split_train', to: 'feat_incongruity', type: 'train' },
  { from: 'split_train', to: 'feat_cleanlab', type: 'train' },

  // Features -> Models
  { from: 'feat_tfidf', to: 'model_svm', type: 'train' },
  { from: 'feat_tfidf', to: 'model_distill', type: 'train' },
  { from: 'split_train', to: 'model_roberta', type: 'train' },
  { from: 'feat_incongruity', to: 'model_logreg', type: 'train' },
  { from: 'split_train', to: 'model_vader', type: 'train' },
  { from: 'split_train', to: 'model_majority', type: 'train' },
  { from: 'split_val', to: 'model_svm', type: 'train', label: 'Platt Calibration' },

  // Models -> Evaluation
  { from: 'model_svm', to: 'eval_heldout', type: 'eval' },
  { from: 'model_distill', to: 'eval_heldout', type: 'eval' },
  { from: 'model_roberta', to: 'eval_heldout', type: 'eval' },
  { from: 'model_logreg', to: 'eval_heldout', type: 'eval' },
  { from: 'model_vader', to: 'eval_heldout', type: 'eval' },
  { from: 'model_majority', to: 'eval_heldout', type: 'eval' },
  // Notice: split_test ONLY connects to eval_heldout! ZERO connection to model fitting!
  { from: 'split_test', to: 'eval_heldout', type: 'eval', label: 'Isolated Test Ground-Truth' },
  { from: 'model_svm', to: 'eval_checklist', type: 'eval' },
  { from: 'feat_phrases', to: 'eval_faithfulness', type: 'eval' },

  // Production Inference Path:
  // User -> UI -> FastAPI -> Lang Validator -> Vectorizer -> SVM -> Rewriter -> UI
  { from: 'ui_analyzer', to: 'api_fastapi', type: 'inference', label: 'HTTP POST' },
  { from: 'api_fastapi', to: 'prep_lang', type: 'inference', label: 'Script Check' },
  { from: 'prep_lang', to: 'feat_tfidf', type: 'inference', label: 'Vectorize' },
  { from: 'feat_tfidf', to: 'model_svm', type: 'inference', label: 'Predict' },
  { from: 'model_svm', to: 'feat_phrases', type: 'inference', label: 'Extract Spans' },
  { from: 'feat_phrases', to: 'api_rewriter', type: 'inference', label: 'Rewrite' },
  { from: 'api_rewriter', to: 'api_fastapi', type: 'inference', label: 'Response Payload' },
  { from: 'api_fastapi', to: 'api_db', type: 'inference', label: 'Persist' },
  { from: 'api_fastapi', to: 'ui_analyzer', type: 'inference', label: 'Render' },
  { from: 'ui_thread', to: 'api_fastapi', type: 'inference' },
  { from: 'api_fastapi', to: 'ui_dashboard', type: 'data' }
];

const LAYER_COLORS: Record<string, { border: string; bg: string; text: string; label: string }> = {
  dataset: { border: '#3B82F6', bg: 'rgba(59, 130, 246, 0.08)', text: '#60A5FA', label: 'Dataset Source' },
  preprocess: { border: '#8B5CF6', bg: 'rgba(139, 92, 246, 0.08)', text: '#A78BFA', label: 'Data Preprocessing' },
  features: { border: '#EAB308', bg: 'rgba(234, 179, 8, 0.08)', text: '#FACC15', label: 'Feature Engineering' },
  model: { border: '#10B981', bg: 'rgba(16, 185, 129, 0.08)', text: '#34D399', label: 'Model Architecture' },
  eval: { border: '#EC4899', bg: 'rgba(236, 72, 153, 0.08)', text: '#F472B6', label: 'Evaluation & Robustness' },
  api: { border: '#06B6D4', bg: 'rgba(6, 182, 212, 0.08)', text: '#22D3EE', label: 'FastAPI Runtime' },
  frontend: { border: '#FFFFFF', bg: 'rgba(255, 255, 255, 0.08)', text: '#FFFFFF', label: 'React Frontend' }
};

const COLUMN_NAMES = [
  '1. Data Sources',
  '2. Ingestion & Splits',
  '3. Feature Extraction',
  '4. Model Zoo',
  '5. Evaluation Suite',
  '6. API & Runtime',
  '7. User Interface'
];

export default function InteractiveArchitectureDiagram() {
  const [filterMode, setFilterMode] = useState<'all' | 'training' | 'inference' | 'models'>('all');
  const [selectedNode, setSelectedNode] = useState<ArchNode | null>(ARCH_NODES.find(n => n.id === 'model_svm') || null);
  const [zoom, setZoom] = useState(0.95);
  const [pan, setPan] = useState({ x: 10, y: 10 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter visible nodes based on filterMode
  const visibleNodes = useMemo(() => {
    if (filterMode === 'all') return ARCH_NODES;
    if (filterMode === 'inference') {
      return ARCH_NODES.filter(n => n.isPrimaryInference || n.layer === 'api' || n.layer === 'frontend');
    }
    if (filterMode === 'training') {
      return ARCH_NODES.filter(n => n.layer !== 'frontend');
    }
    if (filterMode === 'models') {
      return ARCH_NODES.filter(n => n.layer === 'model' || n.id === 'eval_heldout' || n.id === 'feat_tfidf');
    }
    return ARCH_NODES;
  }, [filterMode]);

  const visibleNodeIds = useMemo(() => new Set(visibleNodes.map(n => n.id)), [visibleNodes]);

  // Filter edges based on visible nodes
  const visibleEdges = useMemo(() => {
    return ARCH_EDGES.filter(e => {
      if (!visibleNodeIds.has(e.from) || !visibleNodeIds.has(e.to)) return false;
      if (filterMode === 'inference') return e.type === 'inference';
      if (filterMode === 'training') return e.type === 'train' || e.type === 'data' || e.type === 'leakage_guard';
      return true;
    });
  }, [visibleNodeIds, filterMode]);

  // Pan interaction
  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.arch-node') || (e.target as HTMLElement).closest('button')) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
  };

  const handleMouseUp = () => setIsDragging(false);

  // Node position helper (approximate X/Y coordinate inside canvas)
  const getNodePos = (node: ArchNode) => {
    const colWidth = 240;
    const colGap = 40;
    const rowHeight = 110;
    const rowGap = 20;
    const startX = 40;
    const startY = 60;

    const x = startX + node.col * (colWidth + colGap);
    const y = startY + node.row * (rowHeight + rowGap);
    return { x, y, width: colWidth, height: rowHeight };
  };

  // Node center calculation for SVG bezier arrows
  const getNodeCenter = (nodeId: string): { xRight: number; xLeft: number; yCenter: number } | null => {
    const node = ARCH_NODES.find(n => n.id === nodeId);
    if (!node) return null;
    const pos = getNodePos(node);
    return {
      xRight: pos.x + pos.width,
      xLeft: pos.x,
      yCenter: pos.y + pos.height / 2
    };
  };

  return (
    <div className="w-full flex flex-col space-y-6">
      
      {/* ── Top Control & View Bar ── */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-[#262626] bg-[#0A0A0A]">
        {/* View Mode Tabs */}
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-[#050505] border border-[#1C1C1C]">
          {[
            { id: 'all', label: 'Full System Architecture', icon: Layers },
            { id: 'inference', label: 'Live Inference Pipeline (Production)', icon: Zap },
            { id: 'training', label: 'Training & Offline Pipeline', icon: Sliders },
            { id: 'models', label: 'Model Zoo & Benchmarks', icon: Cpu }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setFilterMode(tab.id as any)}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-semibold transition-all ${
                filterMode === tab.id
                  ? 'bg-[#FFFFFF] text-[#000000] shadow-sm'
                  : 'text-[#858585] hover:text-[#FFFFFF] hover:bg-[#141414]'
              }`}
            >
              <tab.icon size={13} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Zoom & Canvas Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setZoom(z => Math.min(z + 0.15, 1.8))}
            className="p-1.5 rounded-lg border border-[#262626] bg-[#101010] text-[#B3B3B3] hover:text-[#FFFFFF] hover:border-[#404040]"
            title="Zoom In"
            aria-label="Zoom In"
          >
            <ZoomIn size={15} />
          </button>
          <span className="text-[11px] font-mono text-[#858585] min-w-10 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom(z => Math.max(z - 0.15, 0.5))}
            className="p-1.5 rounded-lg border border-[#262626] bg-[#101010] text-[#B3B3B3] hover:text-[#FFFFFF] hover:border-[#404040]"
            title="Zoom Out"
            aria-label="Zoom Out"
          >
            <ZoomOut size={15} />
          </button>
          <button
            onClick={() => { setZoom(0.95); setPan({ x: 10, y: 10 }); }}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-[#262626] bg-[#101010] text-[#B3B3B3] hover:text-[#FFFFFF] text-xs font-medium"
            title="Fit Canvas"
          >
            <Maximize2 size={13} /> Fit
          </button>
          <button
            onClick={() => { setZoom(1.0); setPan({ x: 0, y: 0 }); }}
            className="p-1.5 rounded-lg border border-[#262626] bg-[#101010] text-[#B3B3B3] hover:text-[#FFFFFF]"
            title="Reset Position"
            aria-label="Reset Position"
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      {/* ── Main Interactive Canvas Area ── */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        
        {/* Interactive Diagram Canvas (3 Cols on Desktop) */}
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          className="lg:col-span-3 relative h-[680px] rounded-xl border border-[#262626] bg-[#000000] overflow-hidden select-none cursor-grab active:cursor-grabbing"
          style={{
            backgroundImage: 'radial-gradient(#1A1A1A 1px, transparent 1px)',
            backgroundSize: '24px 24px'
          }}
        >
          {/* Column Header Titles */}
          <div
            className="absolute top-2 left-0 right-0 flex pointer-events-none transition-transform duration-75"
            style={{
              transform: `translate(${pan.x}px, 0px) scale(${zoom})`,
              transformOrigin: '0 0'
            }}
          >
            {COLUMN_NAMES.map((colName, idx) => (
              <div
                key={colName}
                style={{
                  position: 'absolute',
                  left: 40 + idx * 280,
                  width: 240
                }}
                className="text-[11px] font-bold uppercase tracking-wider text-[#858585] border-b border-[#262626] pb-1"
              >
                {colName}
              </div>
            ))}
          </div>

          {/* SVG Connection Layer */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{
              overflow: 'visible'
            }}
          >
            <g
              transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}
              className="transition-transform duration-75"
            >
              <defs>
                <marker
                  id="arrow-normal"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1 L 9 5 L 0 9 z" fill="#404040" />
                </marker>
                <marker
                  id="arrow-inference"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="7"
                  markerHeight="7"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1 L 9 5 L 0 9 z" fill="#FFFFFF" />
                </marker>
                <marker
                  id="arrow-leakage"
                  viewBox="0 0 10 10"
                  refX="8"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1 L 9 5 L 0 9 z" fill="#EF4444" />
                </marker>
              </defs>

              {visibleEdges.map((edge, idx) => {
                const src = getNodeCenter(edge.from);
                const dst = getNodeCenter(edge.to);
                if (!src || !dst) return null;

                const isBackward = dst.xLeft <= src.xRight;
                const startX: number = isBackward ? src.xLeft : src.xRight;
                const endX: number = isBackward ? dst.xRight : dst.xLeft;
                const deltaX = Math.abs(endX - startX) * 0.5;

                const pathData = isBackward
                  ? `M ${startX} ${src.yCenter} C ${startX - 60} ${src.yCenter}, ${endX + 60} ${dst.yCenter}, ${endX} ${dst.yCenter}`
                  : `M ${startX} ${src.yCenter} C ${startX + deltaX} ${src.yCenter}, ${endX - deltaX} ${dst.yCenter}, ${endX} ${dst.yCenter}`;

                const isInferencePath = edge.type === 'inference';
                const isLeakageGuard = edge.type === 'leakage_guard';

                return (
                  <g key={`${edge.from}-${edge.to}-${idx}`}>
                    <path
                      d={pathData}
                      fill="none"
                      stroke={isInferencePath ? '#FFFFFF' : isLeakageGuard ? '#EF4444' : '#2B2B2B'}
                      strokeWidth={isInferencePath ? 2.2 : isLeakageGuard ? 1.8 : 1.2}
                      strokeDasharray={isLeakageGuard ? '4,4' : undefined}
                      markerEnd={isInferencePath ? 'url(#arrow-inference)' : isLeakageGuard ? 'url(#arrow-leakage)' : 'url(#arrow-normal)'}
                      opacity={isInferencePath ? 0.95 : isLeakageGuard ? 0.9 : 0.65}
                    />
                    {edge.label && (
                      <text
                        x={(startX + endX) / 2}
                        y={(src.yCenter + dst.yCenter) / 2 - 6}
                        fill={isLeakageGuard ? '#EF4444' : '#858585'}
                        fontSize="9"
                        fontFamily="ui-monospace, monospace"
                        textAnchor="middle"
                        className="pointer-events-none select-none bg-[#000000]"
                      >
                        {edge.label}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>

          {/* Interactive HTML Nodes */}
          <div
            className="absolute inset-0 transition-transform duration-75"
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: '0 0'
            }}
          >
            {visibleNodes.map(node => {
              const pos = getNodePos(node);
              const isSelected = selectedNode?.id === node.id;
              const isPrimaryInference = node.isPrimaryInference && filterMode === 'inference';
              const layerStyle = LAYER_COLORS[node.layer];

              return (
                <div
                  key={node.id}
                  onClick={() => setSelectedNode(node)}
                  style={{
                    position: 'absolute',
                    left: pos.x,
                    top: pos.y,
                    width: pos.width,
                    height: pos.height
                  }}
                  className={`arch-node group p-3.5 rounded-xl border cursor-pointer transition-all ${
                    isSelected
                      ? 'border-[#FFFFFF] bg-[#171717] ring-1 ring-[#FFFFFF] shadow-2xl'
                      : isPrimaryInference
                      ? 'border-[#FFFFFF] bg-[#121212]'
                      : 'border-[#262626] bg-[#0A0A0A] hover:border-[#444444] hover:bg-[#121212]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-1.5 mb-1.5">
                    <span
                      className="text-[10px] font-mono px-1.5 py-0.5 rounded border"
                      style={{
                        borderColor: layerStyle.border,
                        color: layerStyle.text,
                        backgroundColor: layerStyle.bg
                      }}
                    >
                      {layerStyle.label}
                    </span>
                    {node.status === 'production' && (
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                        Production
                      </span>
                    )}
                    {node.status === 'isolated' && (
                      <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800/60 flex items-center gap-1">
                        <Lock size={9} /> Isolated
                      </span>
                    )}
                  </div>

                  <h4 className="text-xs font-bold text-[#FFFFFF] truncate group-hover:text-white mb-1">
                    {node.name}
                  </h4>

                  <p className="text-[11px] text-[#858585] line-clamp-2 leading-relaxed">
                    {node.description}
                  </p>

                  {/* Metric Chips if available */}
                  {node.metrics && node.metrics.length > 0 && (
                    <div className="flex items-center gap-2 mt-2 pt-1.5 border-t border-[#1C1C1C] text-[10px] font-mono">
                      {node.metrics.slice(0, 2).map((m, i) => (
                        <span key={i} className="text-[#B3B3B3]">
                          {m.label}: <strong className="text-[#FFFFFF]">{m.value}</strong>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Mode Watermark Tag */}
          <div className="absolute bottom-3 left-4 text-[10px] font-mono text-[#525252] pointer-events-none">
            MODE: {filterMode.toUpperCase()} • NODES: {visibleNodes.length} • EDGES: {visibleEdges.length} • DRAG TO PAN
          </div>
        </div>

        {/* Selected Node Details Inspector (1 Col on Desktop) */}
        <div className="p-5 rounded-xl border border-[#262626] bg-[#0A0A0A] flex flex-col justify-between h-[680px] overflow-y-auto">
          {selectedNode ? (
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span
                    className="text-[10px] font-mono uppercase px-2 py-0.5 rounded border"
                    style={{
                      borderColor: LAYER_COLORS[selectedNode.layer].border,
                      color: LAYER_COLORS[selectedNode.layer].text,
                      backgroundColor: LAYER_COLORS[selectedNode.layer].bg
                    }}
                  >
                    {LAYER_COLORS[selectedNode.layer].label}
                  </span>
                  <span className="text-[10px] font-mono text-[#858585]">
                    ID: {selectedNode.id}
                  </span>
                </div>
                <h3 className="text-base font-bold text-[#FFFFFF] leading-snug">
                  {selectedNode.name}
                </h3>
              </div>

              <div className="text-xs text-[#B3B3B3] leading-relaxed">
                {selectedNode.description}
              </div>

              {/* Verified Metrics Grid */}
              {selectedNode.metrics && selectedNode.metrics.length > 0 && (
                <div className="p-3.5 rounded-lg bg-[#101010] border border-[#262626] space-y-2">
                  <span className="text-[11px] font-bold text-[#858585] uppercase tracking-wider block">
                    Verified Performance & Attributes
                  </span>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {selectedNode.metrics.map((m, i) => (
                      <div key={i} className="flex flex-col">
                        <span className="text-[10px] text-[#858585]">{m.label}</span>
                        <span className="font-mono font-bold text-[#FFFFFF]">{m.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Source Implementation */}
              {selectedNode.file && (
                <div className="text-xs">
                  <span className="text-[11px] font-semibold text-[#858585] block mb-1">
                    Implementation File
                  </span>
                  <div className="p-2 rounded bg-[#050505] border border-[#262626] font-mono text-[11px] text-[#FFFFFF] break-all">
                    {selectedNode.file}
                  </div>
                </div>
              )}

              {/* Inputs & Outputs */}
              <div className="space-y-2.5 text-xs">
                <div>
                  <span className="text-[11px] font-semibold text-[#858585] block mb-1">
                    Inputs / Upstream Predecessors
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {selectedNode.inputs.map((inp, idx) => (
                      <span key={idx} className="px-2 py-0.5 rounded bg-[#121212] border border-[#262626] text-[11px] text-[#B3B3B3]">
                        {inp}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <span className="text-[11px] font-semibold text-[#858585] block mb-1">
                    Outputs / Downstream Consumers
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {selectedNode.outputs.map((out, idx) => (
                      <span key={idx} className="px-2 py-0.5 rounded bg-[#121212] border border-[#262626] text-[11px] text-[#FFFFFF]">
                        {out}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Data Leakage Notice if relevant */}
              {selectedNode.id === 'split_test' && (
                <div className="p-3 rounded-lg border border-rose-900/60 bg-rose-950/20 text-xs text-rose-300 space-y-1">
                  <div className="flex items-center gap-1.5 font-bold">
                    <Lock size={13} /> Strict Leakage Guard Verified
                  </div>
                  <p className="text-[11px] text-rose-300/80 leading-relaxed">
                    This test partition has 0 string overlap with the training set and does not feed into model parameter fitting.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center text-[#858585] p-4">
              <Filter size={24} className="mb-2 text-[#404040]" />
              <p className="text-xs">Click any component node in the architecture diagram to inspect verified implementation details.</p>
            </div>
          )}

          <div className="pt-4 border-t border-[#1C1C1C] flex items-center justify-between text-[11px] text-[#858585]">
            <span>Diagram synced with codebase</span>
            <a href="/dashboard" className="text-[#FFFFFF] hover:underline flex items-center gap-1">
              Datasets <ArrowRight size={11} />
            </a>
          </div>
        </div>
      </div>

      {/* ── Architecture Legend & Status Guide ── */}
      <div className="p-4 rounded-xl border border-[#262626] bg-[#0A0A0A] flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-2">
          <span className="font-bold text-[#FFFFFF] uppercase tracking-wider text-[11px]">Legend:</span>
          <div className="flex flex-wrap items-center gap-3">
            {Object.entries(LAYER_COLORS).map(([key, val]) => (
              <div key={key} className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: val.border }} />
                <span className="text-[#B3B3B3] text-[11px]">{val.label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-4 text-[11px] text-[#858585]">
          <div className="flex items-center gap-1.5">
            <span className="w-4 h-0.5 bg-[#FFFFFF] inline-block" />
            <span>Active Inference Flow</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-4 h-0.5 bg-[#404040] inline-block" />
            <span>Training Dependency</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-4 h-0.5 border-t border-dashed border-[#EF4444] inline-block" />
            <span>Leakage Isolation Guard</span>
          </div>
        </div>
      </div>

    </div>
  );
}

import { useState, useRef, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import { ThumbsUp, ThumbsDown, Copy, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { analyzeText } from '../lib/api';
import { mockAnalyze } from '../lib/mockApi';
import { sendFeedback } from '../lib/api';
import { mockFeedback } from '../lib/mockApi';
import { useSettingsStore } from '../store/settings';
import { useHistoryStore } from '../store/history';
import type { AnalyzeResponse, Context, Sensitivity } from '../lib/types';
import { notify } from '../components/Toast';
import ScoreRing from '../components/ScoreRing';
import VerdictPill from '../components/VerdictPill';
import TacticChip from '../components/TacticChip';
import AnnotatedText from '../components/AnnotatedText';
import HighlightList from '../components/HighlightList';
import RewriteCard from '../components/RewriteCard';
import SentenceBars from '../components/SentenceBars';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import SkeletonResult from '../components/SkeletonResult';

const MAX_CHARS = 2000;

const SAMPLES = [
  'Fine. Do whatever you want.',
  'As per my last email, the deadline was Friday.',
  'Thanks for finally getting that done. No worries, I\'m sure you were busy.',
  'Sure, whatever you think is best.',
  'Great job! Honestly I\'m surprised you pulled it off.',
];

const CONTEXTS: { value: Context; label: string }[] = [
  { value: 'workplace', label: 'Workplace email' },
  { value: 'chat', label: 'Chat' },
  { value: 'social', label: 'Social' },
];

const SENSITIVITIES: { value: Sensitivity; label: string }[] = [
  { value: 'lenient', label: 'Lenient' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'strict', label: 'Strict' },
];

export default function Analyzer() {
  const { useMock, context, sensitivity, setContext, setSensitivity } = useSettingsStore();
  const addEntry = useHistoryStore((s) => s.addEntry);

  const [text, setText] = useState('');
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [reportCopied, setReportCopied] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const analyzeFn = useMock ? mockAnalyze : analyzeText;

  const mutation = useMutation({
    mutationFn: () => analyzeFn({ text, context, sensitivity }),
    onSuccess: (data) => {
      setResult(data);
      setFeedbackSent(false);
      addEntry({
        id: data.id,
        timestamp: new Date().toISOString(),
        text,
        context,
        sensitivity,
        result: data,
      });
    },
  });

  const handleAnalyze = useCallback(() => {
    if (!text.trim() || text.length > MAX_CHARS) return;
    mutation.mutate();
  }, [text, mutation]);

  // Ctrl/Cmd + Enter
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleAnalyze();
    }
  };

  const handleUseRewrite = (t: string) => {
    setText(t);
    setResult(null);
    textareaRef.current?.focus();
    setTimeout(() => mutation.mutate(), 50);
  };

  const handleFeedback = async (helpful: boolean) => {
    if (!result || feedbackSent) return;
    try {
      const fb = useMock ? mockFeedback : sendFeedback;
      await fb({ id: result.id, helpful });
      setFeedbackSent(true);
      notify('Thanks for the feedback!');
    } catch {
      notify('Could not send feedback.', 'error');
    }
  };

  const handleCopyReport = () => {
    if (!result) return;
    const report = [
      `PassiveAggression Detector Report`,
      `Score: ${result.score}/100 — ${result.label}`,
      `Tactics: ${result.tactics.map((t) => t.name).join(', ') || 'None'}`,
      '',
      `Original text:`,
      text,
      '',
      `Rewrites:`,
      ...result.rewrites.map((r, i) => `${i + 1}. ${r}`),
    ].join('\n');
    navigator.clipboard.writeText(report).then(() => {
      setReportCopied(true);
      notify('Report copied!');
      setTimeout(() => setReportCopied(false), 2000);
    });
  };

  const sensitivitySliderVal = SENSITIVITIES.findIndex((s) => s.value === sensitivity);

  return (
    <div className="page-container" style={{ paddingTop: 36, paddingBottom: 80 }}>
      {/* Header */}
      <header style={{ marginBottom: 32, textAlign: 'center' }}>
        <h1
          style={{
            fontSize: 'clamp(26px, 4vw, 36px)',
            fontWeight: 800,
            lineHeight: 1.25,
            marginBottom: 10,
            letterSpacing: '-0.03em',
            color: '#FFFFFF',
          }}
        >
          Single Message Analyzer
        </h1>
        <p style={{ fontSize: 15, color: 'var(--text-secondary)', maxWidth: 560, margin: '0 auto', lineHeight: 1.6 }}>
          Surface covert hostility, mock politeness, and passive-aggressive framing with interpretable NLP features.
        </p>
      </header>

      {/* Input card */}
      <section
        className="card"
        aria-label="Message input"
        style={{ marginBottom: 32, display: 'flex', flexDirection: 'column', gap: 20 }}
      >
        {/* Context segmented control */}
        <div>
          <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>
            Communication Context
          </label>
          <div role="group" aria-label="Context" style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {CONTEXTS.map((c) => {
              const active = context === c.value;
              return (
                <button
                  key={c.value}
                  id={`context-${c.value}`}
                  onClick={() => setContext(c.value)}
                  aria-pressed={active}
                  className="btn"
                  style={{
                    padding: '8px 16px',
                    fontSize: 13,
                    fontWeight: active ? 600 : 500,
                    background: active ? '#FFFFFF' : '#101010',
                    color: active ? '#000000' : 'var(--text-secondary)',
                    border: `1px solid ${active ? '#FFFFFF' : '#262626'}`,
                  }}
                >
                  {c.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Textarea */}
        <div style={{ position: 'relative' }}>
          <textarea
            ref={textareaRef}
            id="message-input"
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
            onKeyDown={handleKeyDown}
            placeholder="Paste an email, Slack snippet, or comment here to inspect..."
            rows={7}
            aria-label="Message to analyze"
            aria-describedby="char-counter"
            style={{
              width: '100%',
              padding: '14px 16px',
              fontSize: 14,
              lineHeight: 1.7,
              backgroundColor: '#050505',
              border: '1px solid #262626',
              borderRadius: 8,
              color: '#FFFFFF',
              fontFamily: 'inherit',
            }}
          />
          <span
            id="char-counter"
            aria-live="polite"
            style={{
              position: 'absolute',
              bottom: 12,
              right: 14,
              fontSize: 11,
              color: text.length > MAX_CHARS * 0.9 ? 'var(--hostile)' : 'var(--muted)',
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
            }}
          >
            {text.length}/{MAX_CHARS}
          </span>
        </div>

        {/* Sample chips */}
        <div>
          <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)', display: 'block', marginBottom: 8 }}>
            Try verified benchmark sample:
          </span>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }} role="group" aria-label="Sample messages">
            {SAMPLES.map((s) => (
              <button
                key={s}
                onClick={() => setText(s)}
                className="chip"
                style={{ cursor: 'pointer', fontSize: 12 }}
                aria-label={`Use sample: ${s}`}
              >
                {s.slice(0, 38)}{s.length > 38 ? '…' : ''}
              </button>
            ))}
          </div>
        </div>

        {/* Sensitivity slider */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 6 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label
              htmlFor="sensitivity-slider"
              style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}
            >
              Classification Threshold: <strong style={{ color: '#FFFFFF' }}>{SENSITIVITIES[sensitivitySliderVal]?.label}</strong>
            </label>
            <span style={{ fontSize: 12, color: 'var(--muted)' }}>
              Multiplier: {sensitivitySliderVal === 0 ? '0.8×' : sensitivitySliderVal === 1 ? '1.0×' : '1.2×'}
            </span>
          </div>
          <input
            id="sensitivity-slider"
            type="range"
            min={0}
            max={2}
            step={1}
            value={sensitivitySliderVal}
            onChange={(e) => setSensitivity(SENSITIVITIES[Number(e.target.value)].value)}
            style={{ width: '100%', maxWidth: 320, accentColor: '#FFFFFF' }}
            aria-valuetext={SENSITIVITIES[sensitivitySliderVal]?.label}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', maxWidth: 320, fontSize: 11, color: 'var(--muted)' }}>
            <span>Lenient (Conservative)</span>
            <span>Balanced</span>
            <span>Strict (Sensitive)</span>
          </div>
        </div>

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', paddingTop: 8, borderTop: '1px solid #1C1C1C' }}>
          <button
            id="analyze-btn"
            onClick={handleAnalyze}
            disabled={!text.trim() || mutation.isPending}
            className="btn btn-primary"
            style={{ minWidth: 120 }}
          >
            {mutation.isPending ? 'Analyzing…' : 'Analyze Message'}
          </button>
          <button
            id="clear-btn"
            onClick={() => { setText(''); setResult(null); mutation.reset(); }}
            className="btn btn-ghost"
          >
            Clear
          </button>
          <span style={{ fontSize: 12, color: 'var(--muted)', marginLeft: 'auto' }}>
            Press <kbd style={{ padding: '2px 5px', background: '#171717', border: '1px solid #333', borderRadius: 4, fontSize: 11 }}>Ctrl+Enter</kbd> to analyze
          </span>
        </div>
      </section>

      {/* Results area */}
      <div aria-live="polite" aria-atomic="false">
        {mutation.isPending && <SkeletonResult />}

        {mutation.isError && !mutation.isPending && (
          <ErrorState
            message={(mutation.error as Error).message}
            onRetry={() => mutation.mutate()}
            onSwitchToMock={() => useSettingsStore.setState({ useMock: true })}
          />
        )}

        {!mutation.isPending && !mutation.isError && !result && (
          <EmptyState message="Paste a message to begin." />
        )}

        <AnimatePresence>
          {result && !mutation.isPending && (
            <motion.div
              key={result.id}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.22 }}
              className="card"
              style={{ display: 'flex', flexDirection: 'column', gap: 32 }}
            >
              {/* 1. Score ring + verdict */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                <ScoreRing score={result.score} size={160} />
                <VerdictPill label={result.label} score={result.score} />
              </div>

              {/* 2. Tactics */}
              <div>
                <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>Tactics detected</h2>
                {result.tactics.length > 0 ? (
                  <div
                    role="list"
                    aria-label="Tactics"
                    style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}
                  >
                    {result.tactics.map((t) => (
                      <TacticChip key={t.name} name={t.name} confidence={t.confidence} />
                    ))}
                  </div>
                ) : (
                  <p style={{ fontSize: 14, color: 'var(--muted)' }}>No tactics detected — this message reads as straightforward.</p>
                )}
              </div>

              {/* 3. Annotated text */}
              <AnnotatedText text={text} highlights={result.highlights} />

              {/* 4. Highlight list */}
              <HighlightList highlights={result.highlights} />

              {/* 5. Rewrites */}
              {result.rewrites.length > 0 && (
                <div>
                  <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>Calmer rewrites</h2>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {result.rewrites.map((r, i) => (
                      <RewriteCard key={i} rewrite={r} index={i} onUse={handleUseRewrite} />
                    ))}
                  </div>
                </div>
              )}

              {/* 6. Sentence breakdown */}
              <SentenceBars sentences={result.sentences} />

              {/* 7. Footer row */}
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: 12,
                  paddingTop: 16,
                  borderTop: '1px solid var(--border)',
                }}
              >
                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    onClick={() => handleFeedback(true)}
                    disabled={feedbackSent}
                    className="btn btn-ghost"
                    style={{ padding: '7px 12px' }}
                    aria-label="This verdict was helpful"
                    id="feedback-thumbs-up"
                  >
                    <ThumbsUp size={15} />
                  </button>
                  <button
                    onClick={() => handleFeedback(false)}
                    disabled={feedbackSent}
                    className="btn btn-ghost"
                    style={{ padding: '7px 12px' }}
                    aria-label="This verdict was not helpful"
                    id="feedback-thumbs-down"
                  >
                    <ThumbsDown size={15} />
                  </button>
                  {feedbackSent && (
                    <span style={{ fontSize: 13, color: 'var(--calm)', alignSelf: 'center' }}>
                      Thanks for your feedback!
                    </span>
                  )}
                </div>
                <button
                  onClick={handleCopyReport}
                  className="btn btn-ghost"
                  style={{ padding: '7px 12px', fontSize: 13 }}
                  id="copy-report-btn"
                >
                  {reportCopied ? <Check size={14} /> : <Copy size={14} />}
                  Copy report
                </button>
                <span
                  style={{
                    marginLeft: 'auto',
                    fontSize: 12,
                    color: 'var(--muted)',
                    fontFamily: 'JetBrains Mono, monospace',
                  }}
                >
                  model {result.model_version}, {result.latency_ms} ms
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

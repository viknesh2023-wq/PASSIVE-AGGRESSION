import { useState, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Upload, Users, MessageSquare, Flag, TrendingUp } from 'lucide-react';
import { mockBatch } from '../lib/mockApi';
import { analyzeBatch } from '../lib/api';
import { useSettingsStore } from '../store/settings';
import type { BatchResult } from '../lib/types';
import { parseThreadText, parseCsvText } from '../lib/csv';
import TensionChart from '../components/TensionChart';
import MessageBubble from '../components/MessageBubble';
import Drawer from '../components/Drawer';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import ScoreRing from '../components/ScoreRing';
import VerdictPill from '../components/VerdictPill';
import TacticChip from '../components/TacticChip';
import { scoreToHeat, tacticText } from '../lib/format';

type FilterType = 'all' | 'calm' | 'tense' | 'hostile';

export default function Thread() {
  const { useMock, context, sensitivity } = useSettingsStore();
  const [rawText, setRawText] = useState('');
  const [results, setResults] = useState<BatchResult[]>([]);
  const [filter, setFilter] = useState<FilterType>('all');
  const [selected, setSelected] = useState<BatchResult | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const batchFn = useMock ? mockBatch : analyzeBatch;

  const mutation = useMutation({
    mutationFn: () => {
      const msgs = parseThreadText(rawText);
      if (msgs.length === 0) throw new Error('No messages found. Use the format "Name: message" (one per line).');
      return batchFn({ messages: msgs, context, sensitivity });
    },
    onSuccess: (data) => {
      // Attach sender names
      const msgs = parseThreadText(rawText);
      const enriched: BatchResult[] = data.results.map((r, i) => ({
        ...r,
        sender: msgs[i]?.sender ?? `Speaker ${i + 1}`,
        id: msgs[i]?.text ?? r.id,
      }));
      setResults(enriched);
    },
  });

  const handleFile = async (file: File) => {
    const content = await file.text();
    if (file.name.endsWith('.csv')) {
      const msgs = parseCsvText(content);
      setRawText(msgs.map((m) => `${m.sender}: ${m.text}`).join('\n'));
    } else {
      setRawText(content);
    }
  };

  const filteredResults = results.filter((r) => {
    if (filter === 'all') return true;
    return scoreToHeat(r.score) === filter;
  });

  const mostCommonTactic = results.length > 0
    ? (() => {
        const map = new Map<string, number>();
        for (const r of results) for (const t of r.tactics) map.set(t.name, (map.get(t.name) ?? 0) + 1);
        const max = Math.max(...Array.from(map.values()));
        return Array.from(map.entries()).find(([, v]) => v === max)?.[0] ?? '—';
      })()
    : '—';

  const FILTERS: { value: FilterType; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'calm', label: 'Calm' },
    { value: 'tense', label: 'Tense' },
    { value: 'hostile', label: 'Hostile' },
  ];

  const loadSampleThread = () => {
    setRawText(
      `Alice: Hey Bob, did you get a chance to review the Q3 slide deck? We need it before EOD.\n` +
      `Bob: Thanks for finally checking in. As per my last email, the client already signed off on Friday.\n` +
      `Alice: Great job! Honestly I'm surprised you pulled it off without asking for another extension.\n` +
      `Bob: No worries, I'm sure you were very busy with higher priorities.\n` +
      `Alice: Fine. Do whatever you want with the final handoff.`
    );
  };

  return (
    <div className="page-container" style={{ paddingTop: 36, paddingBottom: 80 }}>
      <header style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 'clamp(26px, 4vw, 36px)', fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 8, color: '#FFFFFF' }}>
          Thread & Conversation Analyzer
        </h1>
        <p style={{ fontSize: 15, color: 'var(--text-secondary)' }}>
          Evaluate dialogue turns, monitor hostility escalation, and isolate passive-aggressive tactics across multiple speakers.
        </p>
      </header>

      <section className="card" style={{ marginBottom: 28, display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault(); setDragging(false);
            const file = e.dataTransfer.files[0];
            if (file) handleFile(file);
          }}
          onClick={() => fileRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload .txt or .csv file"
          onKeyDown={(e) => e.key === 'Enter' && fileRef.current?.click()}
          style={{
            border: `2px dashed ${dragging ? '#FFFFFF' : '#262626'}`,
            borderRadius: 10,
            padding: '24px 20px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'border-color 150ms ease-out',
            background: dragging ? 'rgba(255, 255, 255, 0.04)' : '#050505',
          }}
        >
          <Upload size={24} style={{ color: 'var(--muted)', margin: '0 auto 8px' }} aria-hidden="true" />
          <p style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
            Drop a <strong style={{ color: '#FFFFFF' }}>.txt</strong> or <strong style={{ color: '#FFFFFF' }}>.csv</strong> conversation log, or click to browse
          </p>
          <input
            ref={fileRef}
            type="file"
            accept=".txt,.csv"
            style={{ display: 'none' }}
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            aria-label="Upload thread file"
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Conversation Input (Format: "Speaker: Message")
          </span>
          <button
            onClick={loadSampleThread}
            className="btn btn-ghost"
            style={{ fontSize: 12, padding: '4px 10px' }}
          >
            Load Sample Thread
          </button>
        </div>

        <textarea
          value={rawText}
          onChange={(e) => setRawText(e.target.value)}
          placeholder={'Alice: As per my last email, the deadline was Friday.\nBob: No worries, I\'m sure you were busy.\nAlice: Fine. Do whatever you want.'}
          rows={7}
          style={{
            width: '100%',
            padding: '12px 16px',
            fontSize: 13,
            lineHeight: 1.7,
            backgroundColor: '#050505',
            border: '1px solid #262626',
            borderRadius: 8,
            color: '#FFFFFF',
            fontFamily: 'inherit',
          }}
          aria-label="Thread text — one message per line in the format Name: message"
        />

        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button
            id="analyze-thread-btn"
            onClick={() => mutation.mutate()}
            disabled={!rawText.trim() || mutation.isPending}
            className="btn btn-primary"
            style={{ minWidth: 140 }}
          >
            {mutation.isPending ? 'Analyzing…' : 'Analyze Thread'}
          </button>
          <button
            onClick={() => { setRawText(''); setResults([]); mutation.reset(); }}
            className="btn btn-ghost"
          >
            Clear
          </button>
        </div>

        {mutation.isError && (
          <ErrorState message={(mutation.error as Error).message} onRetry={() => mutation.mutate()} />
        )}
      </section>

      {results.length === 0 && !mutation.isPending && (
        <EmptyState
          icon={<MessageSquare size={48} style={{ color: 'var(--border)' }} />}
          message="Paste a conversation or upload a file to begin analysis."
        />
      )}

      {results.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {/* Summary cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
            {[
              { icon: <Users size={18} />, label: 'Messages', value: results.length },
              { icon: <Flag size={18} />, label: 'Flagged', value: results.filter((r) => r.score >= 30).length },
              {
                icon: <TrendingUp size={18} />, label: 'Highest score',
                value: <span style={{ fontFamily: 'JetBrains Mono, monospace', color: 'var(--hostile)' }}>{Math.max(...results.map((r) => r.score))}</span>
              },
              { icon: <MessageSquare size={18} />, label: 'Top tactic', value: tacticText(mostCommonTactic as any) },
            ].map((card, i) => (
              <div key={i} className="card-raised" style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--muted)', fontSize: 13 }}>
                  {card.icon}
                  {card.label}
                </div>
                <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text)' }}>{card.value}</div>
              </div>
            ))}
          </div>

          {/* Tension timeline */}
          <div className="card">
            <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Tension timeline</h2>
            <TensionChart results={results} />
          </div>

          {/* Filter chips */}
          <div style={{ display: 'flex', gap: 8 }} role="group" aria-label="Filter messages">
            {FILTERS.map((f) => {
              const active = filter === f.value;
              return (
                <button
                  key={f.value}
                  id={`filter-${f.value}`}
                  onClick={() => setFilter(f.value)}
                  className="chip"
                  aria-pressed={active}
                  style={{
                    cursor: 'pointer',
                    padding: '6px 14px',
                    background: active ? '#FFFFFF' : '#101010',
                    color: active ? '#000000' : 'var(--text-secondary)',
                    border: `1px solid ${active ? '#FFFFFF' : '#262626'}`,
                    fontWeight: active ? 600 : 500,
                  }}
                >
                  {f.label}
                </button>
              );
            })}
          </div>

          {/* Chat bubbles */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {filteredResults.map((r, i) => (
              <MessageBubble key={i} result={r} onClick={() => setSelected(r)} />
            ))}
          </div>
        </div>
      )}

      {/* Detail drawer */}
      <Drawer open={!!selected} onClose={() => setSelected(null)} title="Message detail">
        {selected && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            <div style={{ padding: 14, background: 'var(--raised)', borderRadius: 10, fontSize: 14, lineHeight: 1.6 }}>
              <strong style={{ display: 'block', marginBottom: 6, color: 'var(--muted)', fontSize: 12 }}>{selected.sender}</strong>
              {selected.id}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
              <ScoreRing score={selected.score} size={140} />
              <VerdictPill label={selected.label} score={selected.score} />
            </div>
            {selected.tactics.length > 0 && (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {selected.tactics.map((t) => <TacticChip key={t.name} name={t.name} confidence={t.confidence} />)}
              </div>
            )}
          </div>
        )}
      </Drawer>
    </div>
  );
}

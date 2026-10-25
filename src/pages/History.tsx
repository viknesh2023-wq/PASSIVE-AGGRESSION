import { useState } from 'react';
import { Search, Download, Trash2 } from 'lucide-react';
import { useHistoryStore } from '../store/history';
import type { HistoryEntry } from '../lib/types';
import { formatDate, scoreToHeat, heatColor, truncate } from '../lib/format';
import VerdictPill from '../components/VerdictPill';
import Drawer from '../components/Drawer';
import ConfirmDialog from '../components/ConfirmDialog';
import EmptyState from '../components/EmptyState';
import ScoreRing from '../components/ScoreRing';
import TacticChip from '../components/TacticChip';

export default function History() {
  const { entries, removeEntry, clearAll } = useHistoryStore();
  const [search, setSearch] = useState('');
  const [filterVerdict, setFilterVerdict] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'date' | 'score'>('date');
  const [selected, setSelected] = useState<HistoryEntry | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);

  const filtered = entries
    .filter((e) => {
      if (filterVerdict !== 'all' && e.result.label !== filterVerdict) return false;
      if (search && !e.text.toLowerCase().includes(search.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === 'score') return b.result.score - a.result.score;
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    });

  const exportCSV = () => {
    const header = 'Date,Snippet,Verdict,Score,Tactics\n';
    const rows = filtered
      .map((e) =>
        [
          `"${formatDate(e.timestamp)}"`,
          `"${e.text.slice(0, 60).replace(/"/g, '""')}"`,
          e.result.label,
          e.result.score,
          `"${e.result.tactics.map((t) => t.name).join(', ')}"`,
        ].join(',')
      )
      .join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'pa-history.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  const VERDICTS = [
    { value: 'all', label: 'All verdicts' },
    { value: 'neutral', label: 'Neutral' },
    { value: 'passive_aggressive', label: 'Passive-aggressive' },
    { value: 'overt_hostile', label: 'Openly hostile' },
  ];

  return (
    <div className="page-container" style={{ paddingTop: 36, paddingBottom: 80 }}>
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 32 }}>
        <div>
          <h1 style={{ fontSize: 'clamp(26px, 4vw, 36px)', fontWeight: 800, letterSpacing: '-0.03em', color: '#FFFFFF', marginBottom: 6 }}>
            Analysis History & Audit Log
          </h1>
          <p style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
            {entries.length} persistent local evaluation records stored securely in browser storage.
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <button
            onClick={exportCSV}
            disabled={filtered.length === 0}
            className="btn btn-secondary"
            id="export-csv-btn"
          >
            <Download size={15} /> Export CSV
          </button>
          <button
            onClick={() => setConfirmClear(true)}
            disabled={entries.length === 0}
            className="btn btn-ghost"
            id="clear-all-btn"
            style={{ color: '#EF4444', borderColor: 'rgba(239, 68, 68, 0.3)' }}
          >
            <Trash2 size={15} /> Clear All
          </button>
        </div>
      </header>

      {/* Controls */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
        <div style={{ position: 'relative', flex: '1 1 240px' }}>
          <Search size={15} style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--muted)' }} aria-hidden="true" />
          <input
            type="search"
            placeholder="Search verified snippet text…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ width: '100%', padding: '10px 14px 10px 38px', backgroundColor: '#050505', border: '1px solid #262626' }}
            aria-label="Search history"
            id="history-search"
          />
        </div>
        <select
          value={filterVerdict}
          onChange={(e) => setFilterVerdict(e.target.value)}
          style={{ padding: '10px 14px', backgroundColor: '#050505', border: '1px solid #262626' }}
          aria-label="Filter by verdict"
          id="verdict-filter"
        >
          {VERDICTS.map((v) => <option key={v.value} value={v.value}>{v.label}</option>)}
        </select>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as 'date' | 'score')}
          style={{ padding: '10px 14px', backgroundColor: '#050505', border: '1px solid #262626' }}
          aria-label="Sort by"
          id="sort-select"
        >
          <option value="date">Sort by timestamp</option>
          <option value="score">Sort by hostility score</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <EmptyState
          title="No analyses yet"
          message="Analyze a message from the Analyzer page to see your history here."
        />
      ) : (
        <div
          role="table"
          aria-label="Analysis history"
          style={{ display: 'flex', flexDirection: 'column', gap: 2 }}
        >
          <div
            role="row"
            style={{
              display: 'grid',
              gridTemplateColumns: '160px 1fr 180px 80px auto',
              padding: '8px 16px',
              fontSize: 12,
              fontWeight: 600,
              color: 'var(--muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}
          >
            <span role="columnheader">Date</span>
            <span role="columnheader">Snippet</span>
            <span role="columnheader">Verdict</span>
            <span role="columnheader">Score</span>
            <span role="columnheader">Tactics</span>
          </div>

          {filtered.map((entry) => {
            const heat = scoreToHeat(entry.result.score);
            const color = heatColor(heat);
            return (
              <button
                key={entry.id}
                onClick={() => setSelected(entry)}
                role="row"
                aria-label={`Analysis from ${formatDate(entry.timestamp)}, score ${entry.result.score}`}
                style={{
                  all: 'unset',
                  display: 'grid',
                  gridTemplateColumns: '160px 1fr 180px 80px auto',
                  alignItems: 'center',
                  padding: '12px 16px',
                  borderRadius: 10,
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  cursor: 'pointer',
                  gap: 8,
                  transition: 'background 150ms ease-out',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--raised)')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'var(--surface)')}
              >
                <span role="cell" style={{ fontSize: 12, color: 'var(--muted)' }}>{formatDate(entry.timestamp)}</span>
                <span role="cell" style={{ fontSize: 13, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {truncate(entry.text, 80)}
                </span>
                <span role="cell"><VerdictPill label={entry.result.label} score={entry.result.score} /></span>
                <span role="cell" style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 16, fontWeight: 700, color }}>{entry.result.score}</span>
                <span role="cell" style={{ fontSize: 12, color: 'var(--muted)' }}>
                  {entry.result.tactics.length > 0
                    ? entry.result.tactics.slice(0, 2).map((t) => t.name).join(', ')
                    : '—'}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* Detail drawer */}
      <Drawer open={!!selected} onClose={() => setSelected(null)} title="Analysis detail">
        {selected && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <p style={{ fontSize: 13, color: 'var(--muted)' }}>{formatDate(selected.timestamp)}</p>
            <div style={{ padding: 14, background: 'var(--raised)', borderRadius: 10, fontSize: 14, lineHeight: 1.7 }}>
              {selected.text}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
              <ScoreRing score={selected.result.score} size={140} />
              <VerdictPill label={selected.result.label} score={selected.result.score} />
            </div>
            {selected.result.tactics.length > 0 && (
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {selected.result.tactics.map((t) => <TacticChip key={t.name} name={t.name} confidence={t.confidence} />)}
              </div>
            )}
            <button
              onClick={() => { removeEntry(selected.id); setSelected(null); }}
              className="btn btn-ghost"
              style={{ color: 'var(--hostile)', borderColor: 'var(--hostile)', alignSelf: 'flex-start' }}
              id="delete-entry-btn"
            >
              <Trash2 size={15} /> Delete
            </button>
          </div>
        )}
      </Drawer>

      {/* Confirm clear dialog */}
      <ConfirmDialog
        open={confirmClear}
        title="Clear all history"
        message="This will permanently delete all analysis history stored in this browser. This action cannot be undone."
        confirmLabel="Clear all"
        onConfirm={() => { clearAll(); setConfirmClear(false); }}
        onCancel={() => setConfirmClear(false)}
      />
    </div>
  );
}

import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line,
} from 'recharts';
import { useHistoryStore } from '../store/history';
import { tacticText, labelText } from '../lib/format';
import EmptyState from '../components/EmptyState';
import type { Tactic } from '../lib/types';
import { BarChart3 } from 'lucide-react';

const VERDICT_COLORS = {
  neutral: '#22C55E',
  passive_aggressive: '#F59E0B',
  overt_hostile: '#EF4444',
};

export default function Insights() {
  const entries = useHistoryStore((s) => s.entries);

  if (entries.length === 0) {
    return (
      <div className="page-container" style={{ paddingTop: 40 }}>
        <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>Insights</h1>
        <p style={{ fontSize: 14, color: 'var(--muted)', marginBottom: 32 }}>Derived from your local analysis history.</p>
        <EmptyState
          icon={<BarChart3 size={48} style={{ color: 'var(--border)' }} />}
          title="No data yet"
          message="Run some analyses to see aggregate insights, trends, and patterns here."
        />
      </div>
    );
  }

  // Verdict distribution
  const verdictCounts = entries.reduce<Record<string, number>>((acc, e) => {
    acc[e.result.label] = (acc[e.result.label] ?? 0) + 1;
    return acc;
  }, {});
  const verdictData = Object.entries(verdictCounts).map(([name, value]) => ({ name, value }));

  // Tactic counts
  const tacticCounts = new Map<string, number>();
  for (const e of entries) for (const t of e.result.tactics) {
    tacticCounts.set(t.name, (tacticCounts.get(t.name) ?? 0) + 1);
  }
  const tacticData = Array.from(tacticCounts.entries())
    .map(([name, count]) => ({ name: tacticText(name as Tactic), count }))
    .sort((a, b) => b.count - a.count);

  // Average score per day
  const dayMap = new Map<string, number[]>();
  for (const e of entries) {
    const day = e.timestamp.slice(0, 10);
    if (!dayMap.has(day)) dayMap.set(day, []);
    dayMap.get(day)!.push(e.result.score);
  }
  const dailyData = Array.from(dayMap.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, scores]) => ({
      date,
      avg: Math.round(scores.reduce((s, v) => s + v, 0) / scores.length),
    }));

  // Trigger phrases
  const phraseMap = new Map<string, number>();
  for (const e of entries) for (const h of e.result.highlights) {
    phraseMap.set(h.text, (phraseMap.get(h.text) ?? 0) + 1);
  }
  const topPhrases = Array.from(phraseMap.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);

  const tooltipStyle = {
    background: 'var(--raised)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    padding: '8px 12px',
    fontSize: 12,
    color: 'var(--text)',
  };

  return (
    <div className="page-container" style={{ paddingTop: 36, paddingBottom: 80 }}>
      <header style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 'clamp(26px, 4vw, 36px)', fontWeight: 800, letterSpacing: '-0.03em', color: '#FFFFFF', marginBottom: 6 }}>
          Aggregated Analytics & Insights
        </h1>
        <p style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
          Longitudinal distribution, tactical frequency, and hostility trends derived from {entries.length} analyses.
        </p>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 20 }}>
        {/* Verdict donut */}
        <div className="card">
          <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 20 }}>Verdict distribution</h2>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={verdictData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={80}
                paddingAngle={3}
                aria-label="Verdict distribution donut chart"
              >
                {verdictData.map((entry) => (
                  <Cell key={entry.name} fill={VERDICT_COLORS[entry.name as keyof typeof VERDICT_COLORS] ?? '#888'} />
                ))}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} formatter={(v, n) => [v, labelText(n as any)]} />
            </PieChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 12 }}>
            {verdictData.map((d) => (
              <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: VERDICT_COLORS[d.name as keyof typeof VERDICT_COLORS] }} aria-hidden="true" />
                <span style={{ color: 'var(--muted)' }}>{labelText(d.name as any)}</span>
                <span style={{ fontWeight: 700 }}>{d.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Tactic bar */}
        <div className="card">
          <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 20 }}>Tactic counts</h2>
          {tacticData.length === 0 ? (
            <p style={{ fontSize: 14, color: 'var(--muted)' }}>No tactics detected in history.</p>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={tacticData} layout="vertical" margin={{ left: 0, right: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11, fill: 'var(--muted)' }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: 'var(--muted)' }} width={130} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" fill="var(--accent)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Daily average */}
        {dailyData.length > 1 && (
          <div className="card">
            <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 20 }}>Average score per day</h2>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: 'var(--muted)' }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: 'var(--muted)' }} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="avg" stroke="var(--accent)" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Top phrases */}
        {topPhrases.length > 0 && (
          <div className="card">
            <h2 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Most common trigger phrases</h2>
            <ol style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 8 }}>
              {topPhrases.map(([phrase, count], i) => (
                <li key={phrase} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: 'var(--muted)', minWidth: 18 }}>{i + 1}.</span>
                  <span
                    style={{
                      flex: 1,
                      fontSize: 13,
                      fontFamily: 'JetBrains Mono, monospace',
                      background: 'var(--raised)',
                      padding: '3px 10px',
                      borderRadius: 6,
                      color: 'var(--text)',
                    }}
                  >
                    "{phrase}"
                  </span>
                  <span style={{ fontWeight: 700, fontSize: 13, fontFamily: 'JetBrains Mono, monospace', color: 'var(--accent)', minWidth: 28, textAlign: 'right' }}>
                    ×{count}
                  </span>
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}

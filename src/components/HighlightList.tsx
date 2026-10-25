import type { Highlight } from '../lib/types';
import { heatColor, scoreToHeat } from '../lib/format';

interface HighlightListProps {
  highlights: Highlight[];
}

export default function HighlightList({ highlights }: HighlightListProps) {
  if (highlights.length === 0) return null;
  const sorted = [...highlights].sort((a, b) => b.weight - a.weight);
  const maxW = Math.max(...sorted.map((h) => h.weight), 1);

  return (
    <div>
      <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>Why this was flagged</h3>
      <ol style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {sorted.map((h, i) => {
          const heat = scoreToHeat(h.weight * 2);
          const color = heatColor(heat);
          return (
            <li
              key={i}
              style={{
                padding: '10px 14px',
                borderRadius: 10,
                background: 'var(--raised)',
                border: `1px solid var(--border)`,
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span
                  style={{
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 13,
                    color,
                    fontWeight: 600,
                  }}
                >
                  "{h.text}"
                </span>
                <span style={{ fontSize: 11, color: 'var(--muted)' }}>weight {h.weight}</span>
              </div>
              <p style={{ fontSize: 13, color: 'var(--muted)', lineHeight: 1.5 }}>{h.reason}</p>
              <div style={{ height: 3, background: 'var(--border)', borderRadius: 2 }} aria-hidden="true">
                <div
                  style={{
                    width: `${(h.weight / maxW) * 100}%`,
                    height: '100%',
                    background: color,
                    borderRadius: 2,
                    transition: 'width 600ms ease-out',
                  }}
                />
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

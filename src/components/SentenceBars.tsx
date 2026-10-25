import type { SentenceScore } from '../lib/types';
import { heatColor, scoreToHeat } from '../lib/format';

interface SentenceBarsProps {
  sentences: SentenceScore[];
}

export default function SentenceBars({ sentences }: SentenceBarsProps) {
  if (sentences.length === 0) return null;
  return (
    <div>
      <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 10 }}>Sentence breakdown</h3>
      <div
        style={{ display: 'flex', flexDirection: 'column', gap: 4 }}
        role="list"
        aria-label="Sentence scores"
      >
        {sentences.map((s, i) => {
          const heat = scoreToHeat(s.score);
          const color = heatColor(heat);
          return (
            <div
              key={i}
              role="listitem"
              className="tooltip-trigger"
              style={{ position: 'relative' }}
            >
              <div
                style={{
                  height: 10,
                  borderRadius: 4,
                  background: 'var(--raised)',
                  overflow: 'hidden',
                  border: '1px solid var(--border)',
                }}
                aria-label={`Sentence ${i + 1}: score ${s.score}`}
              >
                <div
                  style={{
                    width: `${s.score}%`,
                    height: '100%',
                    background: color,
                    transition: 'width 600ms ease-out',
                    opacity: 0.85,
                  }}
                  aria-hidden="true"
                />
              </div>
              <span className="tooltip-box" role="tooltip">
                <strong style={{ display: 'block', marginBottom: 4, fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color }}>
                  Score: {s.score}
                </strong>
                {s.text}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

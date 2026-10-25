import type { Tactic } from '../lib/types';
import { tacticText } from '../lib/format';

interface TacticChipProps {
  name: Tactic;
  confidence: number; // 0-1
}

export default function TacticChip({ name, confidence }: TacticChipProps) {
  const pct = Math.round(confidence * 100);
  return (
    <div
      className="chip"
      role="listitem"
      aria-label={`${tacticText(name)}: ${pct}% confidence`}
      style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 4, padding: '8px 12px' }}
    >
      <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>{tacticText(name)}</span>
      <div style={{ width: '100%', height: 3, background: 'var(--border)', borderRadius: 2 }}>
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: 'var(--accent)',
            borderRadius: 2,
            transition: 'width 600ms ease-out',
          }}
          aria-hidden="true"
        />
      </div>
      <span style={{ fontSize: 11, color: 'var(--muted)' }}>{pct}% confidence</span>
    </div>
  );
}

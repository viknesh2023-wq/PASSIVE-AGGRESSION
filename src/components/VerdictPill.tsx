import { Shield, AlertTriangle, Zap } from 'lucide-react';
import type { Label } from '../lib/types';
import { labelText, scoreToHeat } from '../lib/format';

interface VerdictPillProps {
  label: Label;
  score: number;
}

export default function VerdictPill({ label, score }: VerdictPillProps) {
  const heat = scoreToHeat(score);
  const color = heat === 'calm' ? 'var(--calm)' : heat === 'tense' ? 'var(--tense)' : 'var(--hostile)';
  const bg = heat === 'calm'
    ? 'var(--calm-bg)'
    : heat === 'tense'
    ? 'var(--tense-bg)'
    : 'var(--hostile-bg)';
  const borderColor = heat === 'calm'
    ? 'var(--calm-border)'
    : heat === 'tense'
    ? 'var(--tense-border)'
    : 'var(--hostile-border)';
  const Icon = heat === 'calm' ? Shield : heat === 'tense' ? AlertTriangle : Zap;

  return (
    <span
      role="status"
      aria-label={`Verdict: ${labelText(label)}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '5px 14px',
        borderRadius: 999,
        border: `1px solid ${borderColor}`,
        background: bg,
        color,
        fontSize: 13,
        fontWeight: 600,
      }}
    >
      <Icon size={14} aria-hidden="true" />
      {labelText(label)}
    </span>
  );
}

import type { BatchResult } from '../lib/types';
import { scoreToHeat, heatColor, labelText } from '../lib/format';

interface MessageBubbleProps {
  result: BatchResult;
  onClick: () => void;
  hidden?: boolean;
}

export default function MessageBubble({ result, onClick, hidden }: MessageBubbleProps) {
  if (hidden) return null;
  const heat = scoreToHeat(result.score);
  const color = heatColor(heat);

  return (
    <button
      onClick={onClick}
      style={{
        all: 'unset',
        cursor: 'pointer',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        maxWidth: '75%',
        alignSelf: 'flex-start',
      }}
      aria-label={`Message from ${result.sender}, score ${result.score}. Click for details.`}
    >
      <span style={{ fontSize: 12, color: 'var(--muted)', fontWeight: 600, paddingLeft: 12 }}>
        {result.sender}
      </span>
      <div
        style={{
          background: 'var(--raised)',
          border: `1px solid ${color}50`,
          borderRadius: '4px 14px 14px 14px',
          padding: '10px 14px',
          fontSize: 14,
          lineHeight: 1.5,
          color: 'var(--text)',
          transition: 'border-color 150ms ease-out',
        }}
      >
        <p>{result.id}</p>
        <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
          <span
            style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 11,
              fontWeight: 600,
              color,
              background: `${color}18`,
              padding: '2px 8px',
              borderRadius: 999,
              border: `1px solid ${color}40`,
            }}
          >
            {result.score}
          </span>
          <span style={{ fontSize: 11, color: 'var(--muted)' }}>{labelText(result.label)}</span>
        </div>
      </div>
    </button>
  );
}

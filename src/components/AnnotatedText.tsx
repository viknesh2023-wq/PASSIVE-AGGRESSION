import { useState } from 'react';
import type { Highlight } from '../lib/types';
import { heatColor, scoreToHeat } from '../lib/format';

interface AnnotatedTextProps {
  text: string;
  highlights: Highlight[];
}

export default function AnnotatedText({ text, highlights }: AnnotatedTextProps) {
  const [showHighlights, setShowHighlights] = useState(true);

  // Build segments
  const segments: { text: string; highlight?: Highlight }[] = [];
  const sorted = [...highlights].sort((a, b) => a.start - b.start);
  let cursor = 0;
  for (const h of sorted) {
    if (h.start > cursor) segments.push({ text: text.slice(cursor, h.start) });
    if (h.start < text.length) {
      segments.push({ text: text.slice(h.start, h.end), highlight: h });
    }
    cursor = h.end;
  }
  if (cursor < text.length) segments.push({ text: text.slice(cursor) });

  const heat = (h: Highlight) => scoreToHeat(h.weight * 2);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
        <h3 style={{ fontSize: 15, fontWeight: 600 }}>Annotated message</h3>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--muted)', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showHighlights}
            onChange={(e) => setShowHighlights(e.target.checked)}
            aria-label="Show highlights"
          />
          Show highlights
        </label>
      </div>
      <p
        style={{
          lineHeight: 1.75,
          fontSize: 14,
          color: 'var(--text)',
          fontFamily: 'inherit',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {segments.map((seg, i) =>
          seg.highlight && showHighlights ? (
            <span
              key={i}
              tabIndex={0}
              className="tooltip-trigger"
              style={{
                textDecoration: `underline 2px ${heatColor(heat(seg.highlight))}`,
                textUnderlineOffset: '3px',
                cursor: 'help',
                borderRadius: 2,
                background: `${heatColor(heat(seg.highlight))}18`,
              }}
              aria-describedby={`hl-tip-${i}`}
            >
              {seg.text}
              <span
                className="tooltip-box"
                id={`hl-tip-${i}`}
                role="tooltip"
              >
                <strong style={{ display: 'block', marginBottom: 4 }}>{seg.highlight.text}</strong>
                {seg.highlight.reason}
                <br />
                <span style={{ color: 'var(--muted)', fontSize: 11 }}>Weight: {seg.highlight.weight}</span>
              </span>
            </span>
          ) : (
            <span key={i}>{seg.text}</span>
          )
        )}
      </p>
    </div>
  );
}

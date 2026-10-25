import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import type { BatchResult } from '../lib/types';

interface TensionChartProps {
  results: BatchResult[];
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: { value: number; payload: { sender: string; text: string } }[];
  label?: number;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: 'var(--raised)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 13,
      maxWidth: 240,
    }}>
      <strong style={{ color: 'var(--accent)', display: 'block', marginBottom: 4 }}>{d.sender}</strong>
      <span style={{ color: 'var(--muted)', fontSize: 12 }}>{d.text.slice(0, 100)}{d.text.length > 100 ? '…' : ''}</span>
      <div style={{ marginTop: 6, fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}>
        Score: <strong style={{ color: payload[0].value >= 61 ? 'var(--hostile)' : payload[0].value >= 30 ? 'var(--tense)' : 'var(--calm)' }}>
          {payload[0].value}
        </strong>
      </div>
    </div>
  );
}

export default function TensionChart({ results }: TensionChartProps) {
  const data = results.map((r, i) => ({
    index: i + 1,
    score: r.score,
    sender: r.sender,
    text: r.id,
  }));

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        {/* Heat bands */}
        <rect x="0" y="0" width="100%" height="30%" fill="rgba(255,111,97,0.06)" />
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis
          dataKey="index"
          tick={{ fontSize: 11, fill: 'var(--muted)', fontFamily: 'JetBrains Mono' }}
          label={{ value: 'Message #', position: 'insideBottom', offset: -4, fill: 'var(--muted)', fontSize: 11 }}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fontSize: 11, fill: 'var(--muted)', fontFamily: 'JetBrains Mono' }}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={30} stroke="var(--tense)" strokeDasharray="4 4" strokeOpacity={0.5} />
        <ReferenceLine y={61} stroke="var(--hostile)" strokeDasharray="4 4" strokeOpacity={0.5} />
        <Line
          type="monotone"
          dataKey="score"
          stroke="var(--accent)"
          strokeWidth={2}
          dot={{ fill: 'var(--accent)', r: 4 }}
          activeDot={{ r: 6 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

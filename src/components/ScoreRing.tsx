import { motion } from 'framer-motion';
import { scoreToHeat, heatColor } from '../lib/format';

interface ScoreRingProps {
  score: number;
  size?: number;
}

export default function ScoreRing({ score, size = 160 }: ScoreRingProps) {
  const heat = scoreToHeat(score);
  const color = heatColor(heat);
  const r = (size / 2) * 0.75;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = Math.PI * r; // half circle
  const offset = circumference * (1 - score / 100);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <svg
        width={size}
        height={size / 2 + 20}
        viewBox={`0 0 ${size} ${size / 2 + 20}`}
        role="img"
        aria-label={`Score: ${score} out of 100`}
      >
        {/* Track */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="var(--border)"
          strokeWidth={12}
          strokeLinecap="round"
        />
        {/* Filled arc */}
        <motion.path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={12}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
        {/* Score number */}
        <text
          x={cx}
          y={cy + 4}
          textAnchor="middle"
          fontFamily="'JetBrains Mono', monospace"
          fontSize={size * 0.2}
          fontWeight={600}
          fill={color}
        >
          {score}
        </text>
        <text
          x={cx}
          y={cy + 18}
          textAnchor="middle"
          fontFamily="'Inter', sans-serif"
          fontSize={10}
          fill="var(--muted)"
        >
          / 100
        </text>
      </svg>
    </div>
  );
}

import { useState } from 'react';
import { Copy, RefreshCw, Check } from 'lucide-react';
import { notify } from './Toast';

interface RewriteCardProps {
  rewrite: string;
  index: number;
  onUse: (text: string) => void;
}

export default function RewriteCard({ rewrite, index, onUse }: RewriteCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(rewrite).then(() => {
      setCopied(true);
      notify('Copied to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div
      style={{
        padding: '14px 16px',
        borderRadius: 10,
        background: 'var(--raised)',
        border: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}
    >
      <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        Rewrite {index + 1}
      </span>
      <p style={{ fontSize: 14, lineHeight: 1.6, color: 'var(--text)' }}>{rewrite}</p>
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={handleCopy}
          className="btn btn-ghost"
          style={{ fontSize: 13, padding: '7px 12px' }}
          aria-label={`Copy rewrite ${index + 1}`}
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
        <button
          onClick={() => onUse(rewrite)}
          className="btn btn-ghost"
          style={{ fontSize: 13, padding: '7px 12px' }}
          aria-label={`Use rewrite ${index + 1} and re-analyze`}
        >
          <RefreshCw size={14} />
          Use this version
        </button>
      </div>
    </div>
  );
}

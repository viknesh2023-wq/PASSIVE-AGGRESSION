export default function SkeletonResult() {
  const shimmer: React.CSSProperties = {
    background: 'linear-gradient(90deg, var(--raised) 25%, var(--border) 50%, var(--raised) 75%)',
    backgroundSize: '200% 100%',
    animation: 'shimmer 1.4s infinite',
    borderRadius: 8,
  };

  const STAGES = ['Cleaning text', 'Parsing sentences', 'Matching phrases', 'Scoring'];

  return (
    <>
      <style>{`
        @keyframes shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
      <div className="card slide-up" aria-busy="true" aria-label="Analyzing text…">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Score ring placeholder */}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <div style={{ ...shimmer, width: 120, height: 70, borderRadius: '60px 60px 0 0' }} />
          </div>

          {/* Stage indicator */}
          <div style={{ display: 'flex', justifyContent: 'center', gap: 8, flexWrap: 'wrap' }}>
            {STAGES.map((stage, i) => (
              <span
                key={stage}
                style={{
                  fontSize: 12,
                  padding: '4px 10px',
                  borderRadius: 999,
                  border: '1px solid var(--border)',
                  color: 'var(--muted)',
                  animation: `fadeIn 300ms ease-out ${i * 250}ms both`,
                }}
              >
                {stage}
              </span>
            ))}
          </div>

          {/* Placeholder lines */}
          {[80, 100, 65, 90, 55].map((w, i) => (
            <div key={i} style={{ ...shimmer, height: 14, width: `${w}%` }} />
          ))}
        </div>
      </div>
    </>
  );
}

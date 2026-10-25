interface EmptyStateProps {
  title?: string;
  message: string;
  icon?: React.ReactNode;
}

export default function EmptyState({ title, message, icon }: EmptyStateProps) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 16,
        padding: '64px 24px',
        textAlign: 'center',
      }}
      role="status"
      aria-live="polite"
    >
      {icon ?? (
        <span
          aria-hidden="true"
          style={{
            fontFamily: 'Instrument Serif, Georgia, serif',
            fontSize: 80,
            color: 'var(--border)',
            lineHeight: 1,
            fontStyle: 'italic',
          }}
        >
          "
        </span>
      )}
      {title && <h3 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)' }}>{title}</h3>}
      <p style={{ fontSize: 15, color: 'var(--muted)', maxWidth: 320, lineHeight: 1.6 }}>{message}</p>
    </div>
  );
}

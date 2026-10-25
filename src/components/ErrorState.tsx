import { AlertCircle, RefreshCw, Wifi } from 'lucide-react';

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
  onSwitchToMock?: () => void;
}

export default function ErrorState({ message, onRetry, onSwitchToMock }: ErrorStateProps) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 16,
        padding: '48px 24px',
        textAlign: 'center',
      }}
      role="alert"
    >
      <AlertCircle size={40} style={{ color: 'var(--hostile)' }} aria-hidden="true" />
      <p style={{ fontSize: 15, color: 'var(--text)', maxWidth: 360, lineHeight: 1.6 }}>{message}</p>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'center' }}>
        {onRetry && (
          <button onClick={onRetry} className="btn btn-primary" id="retry-btn">
            <RefreshCw size={15} />
            Retry
          </button>
        )}
        {onSwitchToMock && (
          <button onClick={onSwitchToMock} className="btn btn-ghost" id="switch-demo-btn">
            <Wifi size={15} />
            Switch to demo data
          </button>
        )}
      </div>
    </div>
  );
}

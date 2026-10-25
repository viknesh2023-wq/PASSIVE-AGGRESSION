import { CheckCircle, XCircle, AlertCircle, X } from 'lucide-react';
import { useEffect, useState } from 'react';

export type ToastType = 'success' | 'error' | 'info';

interface ToastEvent {
  id: number;
  message: string;
  type: ToastType;
}

let _notify: ((msg: string, type?: ToastType) => void) | null = null;

export function notify(message: string, type: ToastType = 'success') {
  _notify?.(message, type);
}

export default function Toast() {
  const [toasts, setToasts] = useState<ToastEvent[]>([]);

  useEffect(() => {
    _notify = (message, type = 'success') => {
      const id = Date.now();
      setToasts((prev) => [...prev, { id, message, type }]);
      setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4000);
    };
    return () => { _notify = null; };
  }, []);

  const remove = (id: number) => setToasts((prev) => prev.filter((t) => t.id !== id));

  const icons = { success: CheckCircle, error: XCircle, info: AlertCircle };

  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      style={{
        position: 'fixed',
        bottom: 24,
        left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        zIndex: 500,
        pointerEvents: 'none',
      }}
    >
      {toasts.map((t) => {
        const Icon = icons[t.type];
        const color = t.type === 'success' ? 'var(--calm)' : t.type === 'error' ? 'var(--hostile)' : 'var(--accent)';
        return (
          <div
            key={t.id}
            role="status"
            className="slide-up"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              padding: '12px 18px',
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              borderRadius: 10,
              fontSize: 14,
              fontWeight: 500,
              color: 'var(--text)',
              boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
              pointerEvents: 'auto',
              minWidth: 240,
            }}
          >
            <Icon size={16} style={{ color, flexShrink: 0 }} />
            <span style={{ flex: 1 }}>{t.message}</span>
            <button
              onClick={() => remove(t.id)}
              aria-label="Dismiss notification"
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)', padding: 2 }}
            >
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}

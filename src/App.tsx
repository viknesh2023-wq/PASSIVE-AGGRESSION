import { useEffect } from 'react';
import { BrowserRouter } from 'react-router-dom';
import { useSettingsStore } from './store/settings';
import NavBar from './components/NavBar';
import AppRoutes from './routes';
import Toast from './components/Toast';

export default function App() {
  const theme = useSettingsStore((s) => s.theme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return (
    <BrowserRouter>
      <div style={{ minHeight: '100dvh', display: 'flex', flexDirection: 'column' }}>
        <NavBar />
        <main style={{ flex: 1, paddingBottom: 48 }}>
          <AppRoutes />
        </main>
        <Toast />
      </div>
    </BrowserRouter>
  );
}

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Context, Sensitivity } from '../lib/types';

interface SettingsState {
  theme: 'dark' | 'light';
  useMock: boolean;
  context: Context;
  sensitivity: Sensitivity;
  setTheme: (t: 'dark' | 'light') => void;
  toggleTheme: () => void;
  setUseMock: (v: boolean) => void;
  setContext: (c: Context) => void;
  setSensitivity: (s: Sensitivity) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      theme: 'dark',
      useMock: import.meta.env.VITE_USE_MOCK !== 'false',
      context: 'workplace',
      sensitivity: 'balanced',
      setTheme: (theme) => set({ theme }),
      toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),
      setUseMock: (useMock) => set({ useMock }),
      setContext: (context) => set({ context }),
      setSensitivity: (sensitivity) => set({ sensitivity }),
    }),
    { name: 'pa-settings' }
  )
);

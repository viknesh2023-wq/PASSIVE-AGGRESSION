# Frontend Audit and Integration Report

**Audit Date:** 2026-10-14  
**Frontend Location:** `d:\python_folder\src`  
**Build Status:** Clean compilation (`tsc -b && vite build` exits code 0)

---

## 1. Application Architecture

| Property | Value |
|---|---|
| **Framework** | React 19.2.8 + Vite 8.3.4 |
| **Language** | TypeScript 6.0.2 (`strict: true`) |
| **Styling** | Tailwind CSS v4 + Custom Theme Tokens (`index.css`) |
| **State Management** | Zustand 5.0.15 (`settingsStore`, `historyStore`) |
| **Data Fetching** | `@tanstack/react-query` 5.104.1 |
| **Charts** | `recharts` 3.10.1 (ResponsiveContainer, PieChart, BarChart, LineChart) |
| **Animations** | `framer-motion` 14.0.0 |
| **Icons** | `lucide-react` 1.53.0 |

---

## 2. Page & Component Matrix

| Route | Component | Live Functionality |
|---|---|---|
| `/` | `src/pages/Analyzer.tsx` | Single text analysis, live score ring gauge, verdict pill, tactic confidence chips, phrase highlighting, sentence tension bars, calmer rewrites, and user feedback buttons. |
| `/thread` | `src/pages/Thread.tsx` | Multi-turn conversation analyzer, participant filters, tension timeline chart, per-turn verdict pills. |
| `/history` | `src/pages/History.tsx` | Persistent SQLite analysis history viewer, search filter, drawer inspector, CSV exporter, clear all. |
| `/insights` | `src/pages/Insights.tsx` | Historical aggregate distribution pie charts, tactic frequency bar charts, average score timelines. |
| `/how-it-works` | `src/pages/HowItWorks.tsx` | Pipeline documentation, classification taxonomy, 6 tactics, and privacy guidelines. |

---

## 3. API Contract Conformance

The frontend API client (`src/lib/api.ts`) communicates directly with `http://localhost:8000/api`:
- `POST /api/analyze` -> Single message payload `{text, context, sensitivity}` -> returns `AnalyzeResponse`
- `POST /api/batch` -> Multi-message payload `{messages: [...], context, sensitivity}` -> returns `BatchResponse`
- `POST /api/feedback` -> User correction `{id, helpful, corrected_label?}` -> returns `{status: "recorded"}`
- `GET /api/history` -> Returns array of `AnalyzeResponse` objects from SQLite
- `GET /api/health` -> System health and active model metadata

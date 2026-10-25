import { CheckCircle, Globe, Cpu, Layers, BarChart2, Lightbulb, AlertTriangle, Lock } from 'lucide-react';

const STAGES = [
  {
    num: 1,
    icon: <Globe size={20} />,
    title: 'Presentation',
    desc: 'Your message arrives in the browser. The frontend validates length, trims whitespace, and packages the text with your chosen context and sensitivity settings.',
  },
  {
    num: 2,
    icon: <Layers size={20} />,
    title: 'API',
    desc: 'A single POST request carries the payload to the FastAPI backend. The API validates input, enforces rate limits, and routes the request to the analysis pipeline.',
  },
  {
    num: 3,
    icon: <CheckCircle size={20} />,
    title: 'NLP preprocessing',
    desc: 'The text is lowercased, tokenised into sentences and words, and punctuation patterns (ellipses, ALL CAPS) are flagged for later scoring.',
  },
  {
    num: 4,
    icon: <Cpu size={20} />,
    title: 'Feature extraction',
    desc: 'A curated phrase bank is matched against the text. Each hit records its position, the matched phrase, the tactic it signals, and a raw weight.',
  },
  {
    num: 5,
    icon: <BarChart2 size={20} />,
    title: 'Classification engine',
    desc: 'Weights are summed and adjusted by the sensitivity multiplier (0.8×, 1.0×, or 1.2×). A 0–100 score is computed and clamped. No generative LLM is involved — every verdict can be explained word by word.',
  },
  {
    num: 6,
    icon: <Lightbulb size={20} />,
    title: 'Decision and explanation',
    desc: 'Bands determine the label: 0–29 Neutral, 30–60 Passive-aggressive, 61–100 Openly hostile. Exact start/end offsets are returned for every matched phrase, and calmer rewrite suggestions are drawn from a curated dictionary.',
  },
];

export default function HowItWorks() {
  return (
    <div className="page-container" style={{ paddingTop: 36, paddingBottom: 80, maxWidth: 840 }}>
      <header style={{ marginBottom: 40 }}>
        <h1 style={{ fontSize: 'clamp(26px, 4vw, 36px)', fontWeight: 800, letterSpacing: '-0.03em', color: '#FFFFFF', marginBottom: 8 }}>
          Methodology & Processing Architecture
        </h1>
        <p style={{ fontSize: 15, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          A deterministic, interpretable NLP pipeline. Every classification is mathematically bound to detected linguistic markers, sentence weights, and sensitivity thresholds.
        </p>
      </header>

      {/* Stepper */}
      <ol
        style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 0 }}
        aria-label="Pipeline stages"
      >
        {STAGES.map((stage, i) => (
          <li
            key={stage.num}
            style={{ display: 'flex', gap: 20 }}
          >
            {/* Timeline */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
              <div
                style={{
                  width: 44,
                  height: 44,
                  borderRadius: 10,
                  background: '#101010',
                  border: '1px solid #262626',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#FFFFFF',
                  flexShrink: 0,
                }}
                aria-hidden="true"
              >
                {stage.icon}
              </div>
              {i < STAGES.length - 1 && (
                <div style={{ width: 1, flex: 1, minHeight: 36, background: '#262626', margin: '4px 0' }} aria-hidden="true" />
              )}
            </div>

            {/* Content */}
            <div style={{ paddingBottom: i < STAGES.length - 1 ? 32 : 0, paddingTop: 6 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <span
                  style={{
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                    fontSize: 11,
                    color: '#FFFFFF',
                    fontWeight: 700,
                    background: '#141414',
                    border: '1px solid #262626',
                    padding: '2px 8px',
                    borderRadius: 4,
                  }}
                >
                  Stage {stage.num}
                </span>
                <h2 style={{ fontSize: 17, fontWeight: 700, color: '#FFFFFF' }}>{stage.title}</h2>
              </div>
              <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.7 }}>{stage.desc}</p>
            </div>
          </li>
        ))}
      </ol>

      {/* System Boundaries card */}
      <div
        className="card"
        style={{ marginTop: 40, border: '1px solid #262626', background: '#0A0A0A' }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <AlertTriangle size={18} style={{ color: '#F59E0B' }} aria-hidden="true" />
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#FFFFFF' }}>Operational Boundaries & Edge Cases</h2>
        </div>
        <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
          This system is scoped strictly to English text with identifiable linguistic markers. Highly nuanced workplace idioms, sarcastic banter among close peers, or non-English colloquialisms may yield edge-case classifications. All outputs are designed as diagnostic heuristics for communication awareness, not psychological attributions.
        </p>
      </div>

      {/* Privacy card */}
      <div className="card" style={{ marginTop: 20, border: '1px solid #262626', background: '#0A0A0A' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <Lock size={18} style={{ color: '#22C55E' }} aria-hidden="true" />
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#FFFFFF' }}>Data Privacy & Security Guarantee</h2>
        </div>
        <p style={{ fontSize: 14, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
          In local mode, inference executes on your loopback FastAPI server (127.0.0.1:8000) with zero external network telemetry. In browser-only mode, computations occur client-side in memory. Inspection logs remain restricted to local IndexedDB/localStorage and can be purged at any time from the History page.
        </p>
      </div>
    </div>
  );
}

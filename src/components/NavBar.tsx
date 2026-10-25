import { NavLink, useLocation } from 'react-router-dom';
import { FlaskConical, MessageSquare, History, BarChart3, HelpCircle, Layers } from 'lucide-react';
import styles from './NavBar.module.css';

const NAV_LINKS = [
  { to: '/', label: 'Analyzer', icon: FlaskConical },
  { to: '/thread', label: 'Thread Analyzer', icon: MessageSquare },
  { to: '/dashboard', label: 'Technical Dashboard', icon: Layers },
  { to: '/history', label: 'History', icon: History },
  { to: '/insights', label: 'Insights', icon: BarChart3 },
  { to: '/how-it-works', label: 'How It Works', icon: HelpCircle },
];

export default function NavBar() {
  const location = useLocation();

  return (
    <>
      {/* Desktop Navigation */}
      <header className={styles.header}>
        <div className={styles.inner}>
          <NavLink to="/" className={styles.brand} aria-label="PassiveAggression Detector home">
            <span className={styles.monogram}>PA</span>
            <span className={styles.brandName}>PassiveAggression Detector</span>
          </NavLink>

          <nav className={styles.nav} aria-label="Main navigation">
            {NAV_LINKS.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) => `${styles.link} ${isActive ? styles.active : ''}`}
              >
                {label}
              </NavLink>
            ))}
          </nav>

          <div className={styles.actions}>
            {/* Demo data badge is completely removed as requested */}
            <span className="text-[11px] font-mono text-[#858585] border border-[#262626] px-2 py-0.5 rounded bg-[#0A0A0A]">
              v2.0 Local
            </span>
          </div>
        </div>
      </header>

      {/* Mobile bottom tab bar */}
      <nav className={styles.mobileNav} aria-label="Mobile navigation">
        {NAV_LINKS.map(({ to, label, icon: Icon }) => {
          const active = to === '/'
            ? location.pathname === '/'
            : location.pathname.startsWith(to);
          return (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={`${styles.mobileLink} ${active ? styles.mobileLinkActive : ''}`}
              aria-label={label}
            >
              <Icon size={18} />
              <span>{label.replace(' Analyzer', '').replace('Technical ', '')}</span>
            </NavLink>
          );
        })}
      </nav>
    </>
  );
}

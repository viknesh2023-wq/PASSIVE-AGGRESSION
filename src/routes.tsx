import { Routes, Route } from 'react-router-dom';
import Analyzer from './pages/Analyzer';
import Thread from './pages/Thread';
import History from './pages/History';
import Insights from './pages/Insights';
import HowItWorks from './pages/HowItWorks';
import ResearchHub from './pages/ResearchHub';

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Analyzer />} />
      <Route path="/thread" element={<Thread />} />
      <Route path="/history" element={<History />} />
      <Route path="/insights" element={<Insights />} />
      <Route path="/dashboard" element={<ResearchHub />} />
      <Route path="/research-hub" element={<ResearchHub />} />
      <Route path="/how-it-works" element={<HowItWorks />} />
    </Routes>
  );
}

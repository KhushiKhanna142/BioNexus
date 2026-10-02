import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppLayout } from '@/components/layout/AppLayout';
import { DashboardPage } from '@/pages/DashboardPage';
import { PatientAnalysisPage } from '@/pages/PatientAnalysisPage';
import { DataExplorerPage } from '@/pages/DataExplorerPage';
import { ModelInsightsPage } from '@/pages/ModelInsightsPage';
import { CohortAnalysisPage } from '@/pages/CohortAnalysisPage';
import { ResearchPage } from '@/pages/ResearchPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { BiomarkersPage } from '@/pages/BiomarkersPage';
import { PathwayAnalysisPage } from '@/pages/PathwayAnalysisPage';
import { GraphExplorerPage } from '@/pages/GraphExplorerPage';
import { CostAccuracyPage } from '@/pages/CostAccuracyPage';
import { NewCasePage } from '@/pages/NewCasePage';
import { LiveCasePage } from '@/pages/LiveCasePage';
import { BatchTriagePage } from '@/pages/BatchTriagePage';
import { HealthProvider } from '@/lib/health';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <HealthProvider>
        <BrowserRouter
          future={{
            v7_startTransition: true,
            v7_relativeSplatPath: true,
          }}
        >
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/new-case" element={<NewCasePage />} />
              <Route path="/live-case/:id" element={<LiveCasePage />} />
              <Route path="/batch-triage" element={<BatchTriagePage />} />
              <Route path="/patient-analysis" element={<PatientAnalysisPage />} />
              <Route path="/graph-explorer" element={<GraphExplorerPage />} />
              <Route path="/data-explorer" element={<DataExplorerPage />} />
              <Route path="/model-insights" element={<ModelInsightsPage />} />
              <Route path="/cost-accuracy" element={<CostAccuracyPage />} />
              <Route path="/cohort-analysis" element={<CohortAnalysisPage />} />
              <Route path="/research" element={<ResearchPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/biomarkers" element={<BiomarkersPage />} />
              <Route path="/pathway-analysis" element={<PathwayAnalysisPage />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </HealthProvider>
    </QueryClientProvider>
  );
}


export default App;

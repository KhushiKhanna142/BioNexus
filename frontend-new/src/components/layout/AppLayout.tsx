import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { useHealth } from '@/lib/health';
import { AlertTriangle } from 'lucide-react';

export function AppLayout() {
  const health = useHealth();
  const live = health.mode === 'live';

  return (
    <div className="min-h-screen">
      <Sidebar />
      <main className="ml-[220px] max-lg:ml-[68px] max-md:ml-0 min-h-screen">
        {!health.loading && !live && (
          <div className="bg-gradient-to-r from-amber-500 to-amber-400 text-black text-sm font-semibold px-4 py-2.5 text-center flex items-center justify-center gap-2 shadow-sm">
            <AlertTriangle className="w-4 h-4" />
            {health.error
              ? `Inference service offline: ${health.error}`
              : 'Model artifacts are not loaded — API is not serving predictions.'}
          </div>
        )}
        <Outlet />
      </main>
    </div>
  );
}

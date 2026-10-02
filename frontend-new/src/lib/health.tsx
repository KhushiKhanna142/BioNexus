import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { api } from '@/api/client';

export type HealthState = {
  loading: boolean;
  mode: string;
  num_patients: number;
  missing_artifacts: string[];
  error?: string;
};

const HealthContext = createContext<HealthState>({
  loading: true,
  mode: 'unloaded',
  num_patients: 0,
  missing_artifacts: [],
});

export function useHealth() {
  return useContext(HealthContext);
}

export function HealthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<HealthState>({
    loading: true,
    mode: 'unloaded',
    num_patients: 0,
    missing_artifacts: [],
  });

  useEffect(() => {
    api.healthCheck()
      .then((h) => setState({
        loading: false,
        mode: h.mode,
        num_patients: h.num_patients,
        missing_artifacts: h.missing_artifacts || [],
      }))
      .catch((err: Error) => setState({
        loading: false,
        mode: 'unloaded',
        num_patients: 0,
        missing_artifacts: [],
        error: err.message,
      }));
  }, []);

  return <HealthContext.Provider value={state}>{children}</HealthContext.Provider>;
}

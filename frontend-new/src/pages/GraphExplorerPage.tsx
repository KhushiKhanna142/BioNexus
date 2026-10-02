import React from 'react';
import { useSearchParams } from 'react-router-dom';
import { PatientGraphExplorer } from '@/components/PatientGraphExplorer';

export function GraphExplorerPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedPatient = searchParams.get('patient') || 'TCGA-AR-A1AR';

  const handleSelectPatient = (pid: string) => {
    setSearchParams({ patient: pid });
  };

  return (
    <div className="w-full">
      <PatientGraphExplorer
        initialPatientId={selectedPatient}
        onSelectPatient={handleSelectPatient}
      />
    </div>
  );
}

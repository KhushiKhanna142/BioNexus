"""
BioNexus Clinical Multi-Omics Agent implementation for MLflow PyFunc / Agent Registry.
Integrates GraphGPS inference, attribution retrieval, and clinical reasoning.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import pandas as pd
import mlflow.pyfunc
from bio_nexus.observability.tracing import trace_clinical_agent_reasoning


class BioNexusClinicalAgent(mlflow.pyfunc.PythonModel):
    """Multi-omics oncology agent registered in MLflow Agent / Model Registry."""

    def __init__(self, agent_name: str = "BioNexusClinicalAgent-v1"):
        self.agent_name = agent_name

    def load_context(self, context: mlflow.pyfunc.PythonModelContext):
        pass

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame | List[Dict[str, Any]] | Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(model_input, pd.DataFrame):
            records = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            records = [model_input]
        else:
            records = model_input

        results = []
        for rec in records:
            patient_id = rec.get("patient_id", "TCGA-3C-AALI")
            session_id = rec.get("session_id")
            
            # Execute traced agent reasoning
            reasoning_res = trace_clinical_agent_reasoning(
                patient_id=patient_id,
                clinical_data=rec,
                session_id=session_id
            )
            results.append(reasoning_res)

        return results

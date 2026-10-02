"""
BioNexus Prompt Registry and Templates.
Manages clinical prompt templates for subtype prediction, multi-omics explanation,
and treatment resistance reasoning.
"""

from __future__ import annotations

from typing import Dict

CLINICAL_SUBTYPE_EXPLANATION_PROMPT = """You are an expert computational oncology assistant analyzing a TCGA-BRCA patient case.

### PATIENT PROFILE
- Patient ID: {patient_id}
- Age: {age}
- Histology: {histology}
- Clinical Receptor Status: ER={er_status}, PR={pr_status}, HER2={her2_status}

### MULTI-OMICS GRAPH-GPS PREDICTION
- Predicted PAM50 Subtype: {predicted_subtype} (Confidence: {confidence:.2%})
- Modality Contributions: RNA={rna_weight:.1%}, miRNA={mirna_weight:.1%}, DNA Methylation={meth_weight:.1%}

### TOP ATTRIBUTED BIOMARKERS
- RNA Biomarkers: {top_rna}
- miRNA Regulators: {top_mirna}
- Methylation Sites: {top_meth}

### TASK
Synthesize a concise, biologically grounded clinical report explaining:
1. Why the GraphGPS multi-omics model assigned this subtype.
2. How the top biomarkers correlate with receptor status and clinical prognosis.
3. Relevant therapeutic pathways (e.g., Endocrine therapy, CDK4/6 inhibitors, Trastuzumab, Chemotherapy).
"""

BIOMARKER_DISCOVERY_PROMPT = """You are a biomarker discovery agent. Evaluate the following multi-omics feature attributions for patient {patient_id}:
RNA: {top_rna}
miRNA: {top_mirna}
Methylation: {top_meth}

Identify any novel cross-modal regulatory axes and return a structured JSON synthesis.
"""

PROMPT_REGISTRY: Dict[str, Dict[str, str]] = {
    "clinical_subtype_explanation": {
        "version": "v1.2",
        "description": "Explains GraphGPS PAM50 multi-omics predictions with biomarker attribution.",
        "template": CLINICAL_SUBTYPE_EXPLANATION_PROMPT,
    },
    "biomarker_discovery": {
        "version": "v1.0",
        "description": "Identifies cross-modal gene-miRNA-methylation regulatory axes.",
        "template": BIOMARKER_DISCOVERY_PROMPT,
    },
}

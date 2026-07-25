"""
Stage 1 — Data Ingestion & Preprocessing.

Owner: ML side (multi-omics preprocessing logic goes here).
MLOps contract: every public function here must be pure (no hidden global
state, no reading env vars mid-function) so it is independently unit-testable
and safely cacheable/DVC-trackable.

Do not put TCGA download credentials or paths here — those belong in
configs/data/*.yaml and are injected at call time.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_omics_layer(path: Path, layer: str) -> pd.DataFrame:
    """Load a single omics layer (expression / methylation / mirna) from disk.

    Args:
        path: path to the raw or cached omics file (parquet/csv).
        layer: one of {"expression", "methylation", "mirna"}.

    Returns:
        DataFrame indexed by patient/sample ID, columns = genes/probes.

    Raises:
        ValueError: if `layer` is not a recognized omics type.
    """
    valid_layers = {"expression", "methylation", "mirna"}
    if layer not in valid_layers:
        raise ValueError(f"Unknown omics layer '{layer}', expected one of {valid_layers}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return df


def align_patients_across_layers(layers: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Keep only patients present in ALL omics layers (inner join on sample ID).

    This is the step that most multi-omics pipelines get quietly wrong —
    if you don't enforce this, downstream graph construction will silently
    misalign patients across modalities.
    """
    common_ids = set.intersection(*(set(df.index) for df in layers.values()))
    if not common_ids:
        raise ValueError("No overlapping patient IDs found across omics layers.")
    return {name: df.loc[sorted(common_ids)] for name, df in layers.items()}

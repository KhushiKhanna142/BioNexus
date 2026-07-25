"""
Model quality gate. This is what CI runs BEFORE promote_model.py --
promotion to Production is conditional on this passing, not just on
training finishing without crashing.

Checks:
1. The Staging model loads without error.
2. It produces output of the expected shape for a dummy input.
3. It beats a trivial baseline (better than random / majority-class guess)
   on a small held-out fixture set.

Extend this as the ML side's interface solidifies -- right now it
documents the *shape* of the gate so CI has something real to enforce
from day one, even before the model itself exists.
"""
import os

import pytest


@pytest.mark.skipif(
    not os.getenv("MLFLOW_TRACKING_URI"),
    reason="Requires MLFLOW_TRACKING_URI -- skipped in local unit-test runs, "
    "runs in CI where the secret is configured.",
)
def test_staging_model_loads():
    import mlflow

    from mosaic.registry.register_model import MODEL_NAME

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Staging")
    assert model is not None


@pytest.mark.skip(reason="Enable once mosaic.fusion + mosaic.models.classifier are implemented")
def test_staging_model_beats_majority_baseline():
    """Placeholder: assert staging model accuracy > majority-class accuracy
    on tests/fixtures/held_out_sample.parquet once that fixture exists."""
    raise NotImplementedError

"""Unit tests for data ingestion contracts (Stage 1)."""
import pandas as pd
import pytest

from mosaic.data.ingestion import align_patients_across_layers


def test_align_patients_keeps_only_common_ids():
    expr = pd.DataFrame({"gene1": [1, 2, 3]}, index=["p1", "p2", "p3"])
    methyl = pd.DataFrame({"probe1": [4, 5]}, index=["p1", "p2"])
    result = align_patients_across_layers({"expression": expr, "methylation": methyl})
    assert set(result["expression"].index) == {"p1", "p2"}
    assert set(result["methylation"].index) == {"p1", "p2"}


def test_align_patients_raises_on_no_overlap():
    expr = pd.DataFrame({"gene1": [1]}, index=["p1"])
    methyl = pd.DataFrame({"probe1": [4]}, index=["p2"])
    with pytest.raises(ValueError):
        align_patients_across_layers({"expression": expr, "methylation": methyl})

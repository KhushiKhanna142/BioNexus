"""Sanity tests for the config system — these should never break silently."""
from mosaic.utils.config import MosaicConfig


def test_default_config_loads():
    cfg = MosaicConfig()
    assert cfg.model.hidden_dim == 256
    assert cfg.training.device in {"cpu", "cuda"}


def test_config_is_frozen():
    cfg = MosaicConfig()
    try:
        cfg.model.hidden_dim = 999
        raised = False
    except Exception:
        raised = True
    assert raised, "Config should be immutable after load — silent mutation risks reproducibility bugs"

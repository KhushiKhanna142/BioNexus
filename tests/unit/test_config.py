"""Sanity tests for the config system — these should never break silently."""
from bio_nexus.utils.config import BioNexusConfig


def test_default_config_loads():
    cfg = BioNexusConfig()
    assert cfg.model.hidden_dim == 256
    assert cfg.training.device in {"cpu", "cuda"}


def test_config_is_frozen():
    cfg = BioNexusConfig()
    try:
        cfg.model.hidden_dim = 999
        raised = False
    except Exception:
        raised = True
    assert raised, "Config should be immutable after load — silent mutation risks reproducibility bugs"

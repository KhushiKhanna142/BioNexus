"""
Model quality gate and MLflow Registry integration tests.
Verifies training tracking, model logging, model registration in Staging,
and promotion to Production.
"""

import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import pytest
import torch
from torch_geometric.data import Data

from bio_nexus.models.classifier import BioNexusClassifier, ClassifierConfig, PAM50_SUBTYPES
from bio_nexus.registry.promote_model import promote_latest_staging_to_production
from bio_nexus.registry.register_model import MODEL_NAME, register_and_stage


@pytest.fixture
def temp_mlflow_env(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "test_mlflow.db"
    uri = f"sqlite:///{db_path}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return uri, tmp_path


def test_mlflow_logging_and_model_registry(temp_mlflow_env):
    uri, tmp_path = temp_mlflow_env

    # 1. Instantiate a small classifier
    model = BioNexusClassifier(
        rna_in_features=16,
        mirna_in_features=16,
        meth_in_features=16,
        cfg=ClassifierConfig(hidden_dim=32),
    )

    exp_name = "test-bionexus-experiment"
    mlflow.set_experiment(exp_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_params({"n_folds": 5, "lr": 1e-3, "architecture": "GraphGPS"})
        mlflow.log_metrics({"accuracy_mean": 0.88, "macro_f1_mean": 0.86})

        # Log dummy artifact
        dummy_metrics = tmp_path / "metrics.json"
        dummy_metrics.write_text(json.dumps({"accuracy_mean": 0.88}))
        mlflow.log_artifact(str(dummy_metrics), artifact_path="evaluation")

        # Log PyTorch Model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
        )

    # 2. Test Registration to Model Registry
    info = {
        "run_id": run_id,
        "model_artifact_path": "model",
        "model_uri": f"runs:/{run_id}/model",
        "tracking_uri": uri,
        "metrics": {"accuracy_mean": 0.88, "macro_f1_mean": 0.86},
    }

    test_model_name = "test-bionexus-classifier"
    version = register_and_stage(test_model_name, info)
    assert version is not None

    # 3. Test Loading Model from MLflow
    loaded_model = mlflow.pytorch.load_model(f"models:/{test_model_name}/Staging")
    assert loaded_model is not None
    loaded_model.eval()

    # 4. Verify forward pass shape
    num_nodes = 10
    x_rna = torch.randn(num_nodes, 16)
    x_mirna = torch.randn(num_nodes, 16)
    x_meth = torch.randn(num_nodes, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32)

    g_rna = Data(x=x_rna, edge_index=edge_index, edge_attr=edge_attr)
    g_mirna = Data(x=x_mirna, edge_index=edge_index, edge_attr=edge_attr)
    g_meth = Data(x=x_meth, edge_index=edge_index, edge_attr=edge_attr)

    with torch.no_grad():
        logits, fused, attn_weights, pool_weights = loaded_model(g_rna, g_mirna, g_meth)

    assert logits.shape == (num_nodes, len(PAM50_SUBTYPES))
    assert fused.shape == (num_nodes, 64)
    assert attn_weights.shape == (num_nodes, 3, 3)
    assert pool_weights.shape == (num_nodes, 3)

    # 5. Test Promotion to Production
    prod_version = promote_latest_staging_to_production(test_model_name)
    assert prod_version == version

    prod_model = mlflow.pytorch.load_model(f"models:/{test_model_name}/Production")
    assert prod_model is not None

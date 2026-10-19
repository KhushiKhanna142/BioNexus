# BioNexus Model Comparison Engine

Provides empirical benchmark evaluation, ranking, and visualization data comparing the BioNexus multi-omics Graph Transformer against classical and deep learning baseline models.

## Structure
- `registry.py`: Canonical definitions of metrics, categories, PAM50 classes, and formatting.
- `data_source.py`: Loading and schema validation for comparison results (`results.json` or MLflow).
- `analysis.py`: Leaderboard computation, metric ranking, radar chart normalization, and trade-off insights.
- `router.py`: FastAPI endpoints for dashboard, summary, and leaderboards.
- `benchmark/`: Benchmark execution pipeline (`run_benchmark.py`).

## Running the Benchmark
```bash
python -m model_comparison.benchmark.run_benchmark \
  --rna data/serving/rna_matrix.parquet \
  --mirna data/serving/mirna_matrix.parquet \
  --meth data/serving/meth_matrix.parquet \
  --labels data/serving/patients.csv \
  --label-col true_pam50 \
  --folds 5 \
  --seed 42 \
  --external bionexus_spec.json
```

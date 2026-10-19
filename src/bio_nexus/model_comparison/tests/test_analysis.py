import json
import unittest
from pathlib import Path

try:
    from bio_nexus.model_comparison import analysis
    from bio_nexus.model_comparison.data_source import DataValidationError, JsonFileSource, parse_dataset
except ImportError:
    from model_comparison import analysis
    from model_comparison.data_source import DataValidationError, JsonFileSource, parse_dataset

_SAMPLE_DATA = Path(__file__).parent.parent / "data" / "results.sample.json"
DATA = _SAMPLE_DATA if _SAMPLE_DATA.exists() else Path(__file__).parent.parent / "data" / "results.json"


def tiny(**over):
    base = {"models": [
        {"id": "bionexus", "name": "B", "metrics": {"accuracy": 0.9, "inference_ms": 30, "causal_ablation": True}},
        {"id": "xgb", "name": "X", "metrics": {"accuracy": 0.8, "inference_ms": 2, "causal_ablation": False}},
    ]}
    base.update(over)
    return parse_dataset(base)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.ds = JsonFileSource(DATA).load()

    def test_direction_and_winners(self):
        boards = {b["metric_id"]: b for b in analysis.leaderboard(list(self.ds.models))}
        self.assertEqual(boards["accuracy"]["winners"], ["bionexus"])
        self.assertNotIn("bionexus", boards["inference_ms"]["winners"])        # lower is better
        self.assertEqual(boards["cross_cohort_drop"]["winners"], ["bionexus"])  # lower is better

    def test_ties_share_rank(self):
        ds = tiny()
        ds.by_id["xgb"].metrics["accuracy"] = 0.9
        b = analysis.rank_metric(ds.models, analysis.METRIC_BY_ID["accuracy"])
        self.assertEqual([e["rank"] for e in b["entries"]], [1, 1])
        self.assertFalse(b["differentiating"])

    def test_missing_values_unranked_and_excluded(self):
        b = analysis.rank_metric(self.ds.models, analysis.METRIC_BY_ID["params_m"])
        xgb = next(e for e in b["entries"] if e["model_id"] == "xgboost")
        self.assertIsNone(xgb["rank"])

    def test_summary_is_computed_not_hardcoded(self):
        sel = list(self.ds.models)
        s = analysis.summarize(analysis.leaderboard(sel), sel)
        self.assertGreater(s["metrics_led"], 0)
        self.assertLess(s["metrics_led"], s["metrics_considered"])  # loses on cost metrics
        cost = next(c for c in s["by_category"] if c["category"] == "cost")
        self.assertEqual(cost["status"], "trails")

    def test_insights_split(self):
        sel = list(self.ds.models)
        ins = analysis.insights(analysis.leaderboard(sel), sel)
        ex = {e["metric_id"] for e in ins["excels"]}
        lg = {e["metric_id"] for e in ins["lags"]}
        self.assertIn("accuracy", ex)
        self.assertIn("causal_ablation", ex)
        self.assertIn("inference_ms", lg)
        self.assertFalse(ex & lg)

    def test_minority_f1_derived(self):
        self.assertAlmostEqual(self.ds.by_id["bionexus"].metrics["minority_f1"], 0.88)

    def test_filter_always_keeps_focus(self):
        sel = analysis.select_models(self.ds, ["xgboost"])
        self.assertEqual({m.id for m in sel}, {"bionexus", "xgboost"})

    def test_unknown_model_rejected(self):
        with self.assertRaises(analysis.ComparisonError):
            analysis.select_models(self.ds, ["nope"])
        with self.assertRaises(analysis.ComparisonError):
            analysis.leaderboard(list(self.ds.models), "bogus")

    def test_radar_bounds_and_inversion(self):
        r = analysis.radar(list(self.ds.models))
        by = {s["model_id"]: s["values"] for s in r["series"]}
        for v in by.values():
            for score in v.values():
                if score is not None:
                    self.assertTrue(0 <= score <= 100)
        self.assertEqual(by["bionexus"]["accuracy"], 100.0)
        self.assertEqual(by["bionexus"]["generalization"], 100.0)   # lowest drop -> best
        self.assertLess(by["bionexus"]["speed"], by["xgboost"]["speed"])

    def test_per_class(self):
        pc = analysis.per_class(list(self.ds.models))
        self.assertEqual(len(pc["classes"]), 5)
        normal = next(c for c in pc["summary"] if c["class"] == "Normal")
        self.assertTrue(normal["is_minority"])
        self.assertGreater(normal["focus_advantage_pp"], 0)

    def test_dashboard_is_json_serialisable(self):
        json.dumps(analysis.build_dashboard(self.ds, ["mogonet", "xgboost"]))

    def test_validation(self):
        for bad in (
            {"models": []},
            {"models": [{"id": "a", "name": "A", "metrics": {"nonsense": 1}}]},
            {"models": [{"id": "a", "name": "A", "metrics": {"accuracy": "high"}}]},
            {"models": [{"id": "a", "name": "A", "metrics": {"causal_ablation": 1}}]},
            {"models": [{"id": "a", "name": "A", "per_class_f1": {"LumA": 1.4}}]},
            {"models": [{"id": "a", "name": "A"}, {"id": "a", "name": "B"}]},
        ):
            with self.assertRaises(DataValidationError, msg=str(bad)):
                parse_dataset(bad)


if __name__ == "__main__":
    unittest.main()

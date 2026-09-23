from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "bench" / "results" / "m0"
BENCHMARK_KEYS = {
    "schema_version",
    "provenance",
    "correctness",
    "engine",
    "protocol",
    "trials",
    "summary",
}


class ResultTests(unittest.TestCase):
    def test_tracked_results_are_redacted_and_structured(self) -> None:
        files = sorted(RESULT_ROOT.rglob("*.json"))
        self.assertGreater(len(files), 20)
        for path in files:
            raw = path.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", raw, path.name)
            self.assertNotIn("doyeon@", raw, path.name)
            data = json.loads(raw)
            if BENCHMARK_KEYS.issubset(data):
                self.assertEqual(data["schema_version"], 1, path.name)
                self.assertIn(data["correctness"]["status"], {
                    "not-compared",
                    "matched",
                    "mismatched",
                    "reference",
                }, path.name)
                self.assertIn(data["protocol"]["semantic_mode"], {
                    "exact",
                    "approximate",
                    "baseline",
                }, path.name)
                self.assertGreater(len(data["provenance"]["checkpoint_manifest_sha256"]), 0)

    def test_kept_optimization_has_matched_greedy_ids(self) -> None:
        path = RESULT_ROOT / "mlx-lm-current-fused-routed-gate-up-1k-64-greedy.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["correctness"]["status"], "matched")
        self.assertEqual(data["model_load"]["fused_routed_gate_up_layers"], 40)

    def test_invalid_profiler_is_marked(self) -> None:
        path = (
            RESULT_ROOT
            / "invalid"
            / "mlx-component-profile-8bit-lazy-unevaluated.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(data["valid"])
        self.assertIn("mx.eval", data["invalid_reason"])


if __name__ == "__main__":
    unittest.main()

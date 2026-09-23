from __future__ import annotations

import unittest

from runnel_bench.workload import WORKLOADS, get_workload


class WorkloadTests(unittest.TestCase):
    def test_manifest_digests_are_stable(self) -> None:
        first = get_workload("python-multi-file-v1")
        second = WORKLOADS["python-multi-file-v1"]
        self.assertEqual(first.digest(), second.digest())
        self.assertEqual(
            first.digest(),
            "3b0025b132647462ee07755377ba1882f9d0bac978d8dc2af67ed761215b5879",
        )

    def test_unknown_workload_is_explicit(self) -> None:
        with self.assertRaises(KeyError):
            get_workload("missing")


if __name__ == "__main__":
    unittest.main()

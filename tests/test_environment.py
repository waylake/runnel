from __future__ import annotations

import unittest

from runnel_bench.environment import _scrub


class EnvironmentTests(unittest.TestCase):
    def test_scrub_removes_machine_identifiers_recursively(self) -> None:
        value = {
            "serial_number": "private",
            "platform_UUID": "private",
            "model_number": "private",
            "_spdisplays_display-serial-number": "private",
            "nested": [
                {
                    "machine_name": "private",
                    "chip_type": "Apple M1 Max",
                    "power": "InternalBattery-0 (id=12345)",
                }
            ],
            "safe": 32,
        }
        self.assertEqual(
            _scrub(value),
            {
                "nested": [
                    {
                        "chip_type": "Apple M1 Max",
                        "power": "InternalBattery-0 (id=<redacted>)",
                    }
                ],
                "safe": 32,
            },
        )


if __name__ == "__main__":
    unittest.main()

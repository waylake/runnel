from __future__ import annotations

import importlib
import unittest


class RuntimePackageTests(unittest.TestCase):
    def test_package_imports_without_mlx_installed(self) -> None:
        package = importlib.import_module("runnel")
        self.assertEqual(package.__version__, "0.1.0.dev0")

    def test_cli_parser_is_available(self) -> None:
        cli = importlib.import_module("runnel.cli")
        args = cli.build_parser().parse_args(
            [
                "generate",
                "--model",
                "/tmp/model",
                "--prompt",
                "hello",
                "--stock-mlp",
            ]
        )
        self.assertTrue(args.stock_mlp)


if __name__ == "__main__":
    unittest.main()

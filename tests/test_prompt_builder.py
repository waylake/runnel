from __future__ import annotations

import unittest

from runnel_bench.prompt_builder import build_prompt


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(token) for token in token_ids)


class PromptBuilderTests(unittest.TestCase):
    def test_builds_exact_stable_token_count(self) -> None:
        result = build_prompt(
            CharTokenizer(),
            "python-multi-file-v1",
            1024,
            enable_thinking=False,
        )
        self.assertEqual(result["target_prompt_tokens"], 1024)
        self.assertEqual(result["actual_prompt_tokens"], 1024)
        self.assertEqual(len(result["token_ids"]), 1024)
        self.assertFalse(result["enable_thinking"])
        self.assertIn("<think>\n\n</think>", result["text"])


if __name__ == "__main__":
    unittest.main()

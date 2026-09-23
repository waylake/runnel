from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from runnel_bench.runner import run_stream_once


class SSEHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        assert request["temperature"] == 0.0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            {"choices": [{"delta": {"content": "A"}}]},
            {"choices": [{"delta": {"content": "B"}}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: object) -> None:
        return


class RunnerTests(unittest.TestCase):
    def test_stream_collects_usage_and_output_digest(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), SSEHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = run_stream_once(
                endpoint=f"http://127.0.0.1:{server.server_port}/v1/completions",
                model="test",
                prompt="abc",
                expected_prompt_tokens=3,
                max_tokens=2,
                temperature=0.0,
                api_key_env=None,
                timeout=5.0,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["output_bytes"], 2)
        self.assertEqual(result["stream_event_count"], 3)
        self.assertIsNotNone(result["ttft_s"])


if __name__ == "__main__":
    unittest.main()

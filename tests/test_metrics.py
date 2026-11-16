"""Check summary calculations with two small examples."""

import unittest

from inference_server.metrics import summarize_http_results


class MetricsTests(unittest.TestCase):
    def test_throughput_latency_errors_and_queue_wait(self):
        results = [
            {"success": True, "latency_ms": 100, "queue_wait_ms": 0},
            {"success": True, "latency_ms": 300, "queue_wait_ms": 100},
            {"success": False, "latency_ms": 1, "error": "Full", "error_type": "rejected"},
        ]
        summary = summarize_http_results(results, elapsed_seconds=2)

        self.assertEqual(summary["requests_per_second"], 1)
        self.assertEqual(summary["p95_latency_ms"], 290)
        self.assertEqual(round(summary["error_rate_percent"], 2), 33.33)
        self.assertEqual(summary["average_queue_wait_ms"], 50)

    def test_simulation_leaves_token_measurements_unavailable(self):
        results = [{"success": True, "latency_ms": 2000}]
        summary = summarize_http_results(results, elapsed_seconds=2)

        self.assertIsNone(summary["output_tokens_per_second"])
        self.assertIsNone(summary["average_generation_ttft_ms"])

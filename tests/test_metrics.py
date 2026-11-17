"""Check summary calculations with two small examples."""

import unittest

from inference_server.metrics import EngineMetrics, summarize_http_results


class MetricsTests(unittest.TestCase):
    def test_throughput_latency_errors_and_queue_wait(self):
        results = [
            {"success": True, "latency_ms": 100, "queue_wait_ms": 0, "client_ttft_ms": 20, "text_chunk_intervals_ms": [10, 30], "generation_tpot_ms": 10},
            {"success": True, "latency_ms": 300, "queue_wait_ms": 100, "client_ttft_ms": 50, "text_chunk_intervals_ms": [50], "generation_tpot_ms": 30},
            {"success": False, "latency_ms": 1, "error": "Full", "error_type": "rejected"},
        ]
        summary = summarize_http_results(results, elapsed_seconds=2)

        self.assertEqual(summary["requests_per_second"], 1)
        self.assertEqual(summary["p95_latency_ms"], 290)
        self.assertEqual(round(summary["error_rate_percent"], 2), 33.33)
        self.assertEqual(summary["average_queue_wait_ms"], 50)
        self.assertEqual(summary["average_client_ttft_ms"], 35)
        self.assertEqual(summary["average_text_chunk_interval_ms"], 30)
        self.assertEqual(summary["average_generation_tpot_ms"], 20)

    def test_missing_measurements_are_not_estimated(self):
        results = [{"success": True, "latency_ms": 2000}]
        summary = summarize_http_results(results, elapsed_seconds=2)

        self.assertIsNone(summary["output_tokens_per_second"])
        self.assertIsNone(summary["average_generation_ttft_ms"])
        self.assertIsNone(summary["average_client_ttft_ms"])
        self.assertIsNone(summary["average_generation_tpot_ms"])

    def test_engine_peaks_and_forward_sizes_are_observed(self):
        metrics = EngineMetrics()
        metrics.record_queue(20)
        metrics.record_queue(0)
        metrics.record_active(4, 128)
        metrics.record_active(0, 0)
        metrics.record_forward("prefill", 4)
        metrics.record_forward("prefill", 1)
        metrics.record_forward("decode", 2)
        result = metrics.snapshot()
        self.assertEqual((result["peak_queued_requests"], result["peak_active_requests"], result["peak_reserved_cache_tokens"]), (20, 4, 128))
        self.assertEqual((result["mean_prefill_batch_size"], result["mean_decode_batch_size"]), (2.5, 2))
        self.assertEqual(result["reserved_cache_tokens"], 0)

"""Check HTTP measurements without loading a model or starting the real server."""

import threading
import time
import unittest
from unittest.mock import patch

import httpx

import benchmark


class BenchmarkTests(unittest.TestCase):
    def test_summary_counts_successes_and_failures_separately(self):
        results = [
            {"success": True, "latency_ms": 100, "ttft_ms": 10, "output_tokens": 10},
            {"success": False, "latency_ms": 500, "error": "Server error"},
            {"success": True, "latency_ms": 300, "ttft_ms": 30, "output_tokens": 6},
        ]

        summary = benchmark.summarize_http_results(results, elapsed_seconds=2)

        self.assertEqual(summary["total_requests"], 3)
        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["average_latency_ms"], 200)
        self.assertEqual(summary["average_generation_ttft_ms"], 20)
        self.assertEqual(summary["total_output_tokens"], 16)
        self.assertEqual(summary["output_tokens_per_second"], 8)
        self.assertEqual(summary["errors"], ["Server error"])

    def test_summary_handles_no_successful_requests(self):
        results = [{"success": False, "latency_ms": 100, "error": "Timeout"}]
        summary = benchmark.summarize_http_results(results, elapsed_seconds=1)

        self.assertIsNone(summary["average_latency_ms"])
        self.assertIsNone(summary["average_generation_ttft_ms"])
        self.assertEqual(summary["output_tokens_per_second"], 0)

    def test_http_benchmark_overlaps_requests_and_excludes_warmup(self):
        active_requests = 0
        peak_requests = 0
        total_requests = 0
        lock = threading.Lock()

        def handle_request(request):
            nonlocal active_requests, peak_requests, total_requests
            with lock:
                active_requests += 1
                total_requests += 1
                peak_requests = max(peak_requests, active_requests)

            # Give another worker time to enter while this request is running.
            time.sleep(0.02)

            with lock:
                active_requests -= 1
            return httpx.Response(200, json={"output_tokens": 4, "ttft_ms": 2})

        # MockTransport gives the real HTTP client a local response function.
        client = httpx.Client(
            base_url=benchmark.SERVER_URL,
            transport=httpx.MockTransport(handle_request),
        )
        with patch("benchmark.httpx.Client", return_value=client):
            summary = benchmark.run_http_benchmark(concurrency=2)

        self.assertEqual(peak_requests, 2)
        self.assertEqual(total_requests, benchmark.REQUEST_COUNT + 1)
        self.assertEqual(summary["total_requests"], benchmark.REQUEST_COUNT)
        self.assertEqual(summary["total_output_tokens"], benchmark.REQUEST_COUNT * 4)
        self.assertEqual(summary["concurrency"], 2)
        self.assertGreater(summary["elapsed_seconds"], 0)

    def test_server_errors_and_timeouts_are_returned_as_failures(self):
        def server_error(request):
            return httpx.Response(500)

        def timeout(request):
            raise httpx.ReadTimeout("Request timed out", request=request)

        for handler in [server_error, timeout]:
            with self.subTest(handler=handler.__name__):
                with httpx.Client(
                    base_url=benchmark.SERVER_URL,
                    transport=httpx.MockTransport(handler),
                ) as client:
                    result = benchmark.send_request(client, "hello")

                self.assertFalse(result["success"])
                self.assertTrue(result["error"])
                self.assertGreaterEqual(result["latency_ms"], 0)

    def test_failed_warmup_stops_before_the_measured_requests(self):
        total_requests = 0

        def handle_request(request):
            nonlocal total_requests
            total_requests += 1
            return httpx.Response(503)

        client = httpx.Client(
            base_url=benchmark.SERVER_URL,
            transport=httpx.MockTransport(handle_request),
        )
        with patch("benchmark.httpx.Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "Warmup failed"):
                benchmark.run_http_benchmark(concurrency=2)

        self.assertEqual(total_requests, 1)


if __name__ == "__main__":
    unittest.main()

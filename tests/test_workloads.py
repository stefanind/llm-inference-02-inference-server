"""Check workload sizes, mixed lengths, and arrival patterns."""

import unittest

from inference_server.workloads import make_w1, make_w2, make_w3, make_w4, make_w5, make_w6


class WorkloadTests(unittest.TestCase):
    def test_workload_sizes_and_mixed_prompt_groups(self):
        mixed = make_w4("short", "long")
        workloads = [make_w1(), make_w2(), make_w3(), mixed, make_w5(), make_w6(1)]
        self.assertEqual([len(requests) for requests in workloads], [1, 20, 20, 20, 400, 30])
        self.assertEqual(
            [(request.prompt, request.max_new_tokens) for request in mixed],
            [("short", 32)] * 5 + [("short", 256)] * 5 + [("long", 32)] * 5 + [("long", 256)] * 5,
        )

    def test_arrival_patterns(self):
        self.assertTrue(all(request.arrival_ms == 0 for request in make_w5()))
        self.assertEqual(make_w3(seed=42), make_w3(seed=42))
        self.assertEqual(
            [request.arrival_ms for request in make_w6(2, duration_seconds=2)],
            [0, 500, 1000, 1500],
        )

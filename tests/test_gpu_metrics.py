"""Check aggregation and unavailable GPU readings without requiring a GPU."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from inference_server.gpu_metrics import GpuSampler, MIB, TorchMemoryMetrics


class GpuMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_sampler_records_observed_utilization_and_memory_peak(self):
        nvml = SimpleNamespace(
            NVMLError=RuntimeError, nvmlInit=Mock(), nvmlShutdown=Mock(),
            nvmlDeviceGetHandleByUUID=Mock(return_value="handle"),
            nvmlDeviceGetUUID=Mock(return_value="GPU-test"),
            nvmlDeviceGetMemoryInfo=Mock(side_effect=[SimpleNamespace(used=value * MIB) for value in (100, 120, 140, 110)]),
            nvmlDeviceGetUtilizationRates=Mock(side_effect=[SimpleNamespace(gpu=value) for value in (40, 80)]),
        )
        with patch.dict(sys.modules, {"pynvml": nvml}):
            async with GpuSampler("GPU-test") as sampler:
                # Fixed readings keep this arithmetic check independent of timers.
                sampler.sample()
                sampler.sample()
        result = sampler.snapshot()
        self.assertEqual(result["average_gpu_utilization_percent"], 60)
        self.assertEqual((result["baseline_device_memory_mib"], result["sampled_peak_device_memory_mib"]), (100, 140))
        self.assertEqual((result["gpu_utilization_sample_count"], result["device_memory_sample_count"]), (2, 4))
        nvml.nvmlShutdown.assert_called_once()

    async def test_missing_bindings_or_driver_leave_values_unknown(self):
        no_driver = SimpleNamespace(NVMLError=RuntimeError, nvmlInit=Mock(side_effect=RuntimeError("No driver")))
        for module in (None, no_driver):
            with patch.dict(sys.modules, {"pynvml": module}):
                async with GpuSampler("GPU-test") as sampler:
                    pass
            result = sampler.snapshot()
            self.assertIsNone(result["average_gpu_utilization_percent"])
            self.assertIsNone(result["sampled_peak_device_memory_mib"])
            self.assertTrue(result["errors"])

    def test_torch_peak_includes_the_model_baseline(self):
        memory = TorchMemoryMetrics(torch.device("cuda:0"))
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.reset_peak_memory_stats") as reset,
            patch("torch.cuda.memory_allocated", return_value=100 * MIB),
            patch("torch.cuda.max_memory_allocated", return_value=150 * MIB),
        ):
            memory.reset()
            result = memory.snapshot(synchronize=True)
        reset.assert_called_once_with(torch.device("cuda:0"))
        self.assertEqual((result["baseline_allocated_gpu_memory_mib"], result["peak_allocated_gpu_memory_mib"]), (100, 150))

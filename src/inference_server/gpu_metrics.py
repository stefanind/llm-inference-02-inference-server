"""GPU measurements. Importing this module does not load PyTorch or initialize CUDA.

Run the workload client on the GPU host to use NVML. UUID selection prevents
sampling an unrelated GPU when the client runs elsewhere. Device VRAM covers
all processes on that GPU; use a dedicated GPU for comparisons.
"""

import asyncio
import time

MIB = 1024 * 1024


class GpuSampler:
    """Sample NVML every 200 ms around client calls, for either server backend.

    Utilization is the mean of available NVML readings, each of which covers a
    hardware sampling window. Short runs can have zero utilization samples.
    Memory includes baseline/end readings; its sampled peak can miss spikes.
    Initialization and cleanup are outside the client experiment timer.
    """
    def __init__(self, gpu_uuid, interval_ms=200):
        self.requested_uuid = gpu_uuid
        self.gpu_uuid = None
        self.interval_ms = interval_ms
        self.nvml = None
        self.initialized = False
        self.handle = None
        self.baseline_memory = None
        self.peak_memory = None
        self.memory_samples = 0
        self.utilization_total = 0
        self.utilization_samples = 0
        self.errors = []
        self.stopped = asyncio.Event()
        self.task = None
        self.started_at = None
        self.elapsed_seconds = None

    def sample(self, include_utilization=True):
        if self.handle is None:
            return
        try:
            used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
            if used is None:
                self.record_error("VRAM reading unavailable")
            else:
                used /= MIB
                self.peak_memory = used if self.peak_memory is None else max(self.peak_memory, used)
                self.memory_samples += 1
        except self.nvml.NVMLError as error:
            self.record_error(f"VRAM: {error}")
        if include_utilization:
            try:
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
                if utilization is None or not 0 <= utilization <= 100:
                    self.record_error("Utilization reading unavailable")
                else:
                    self.utilization_total += utilization
                    self.utilization_samples += 1
            except self.nvml.NVMLError as error:
                self.record_error(f"Utilization: {error}")

    def record_error(self, message):
        if message not in self.errors:
            self.errors.append(message)

    async def poll(self):
        while not self.stopped.is_set():
            try:
                await asyncio.wait_for(self.stopped.wait(), timeout=self.interval_ms / 1000)
            except asyncio.TimeoutError:
                self.sample()

    async def __aenter__(self):
        if not isinstance(self.requested_uuid, str) or not self.requested_uuid:
            self.record_error("Server GPU UUID unavailable; supply --gpu-uuid when running on the GPU host")
            return self
        try:
            import pynvml
        except ImportError:
            self.record_error("Install nvidia-ml-py on the GPU host for NVML sampling")
            return self
        self.nvml = pynvml
        try:
            pynvml.nvmlInit()
            self.initialized = True
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(self.requested_uuid)
            uuid = pynvml.nvmlDeviceGetUUID(self.handle)
            self.gpu_uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
            self.sample(include_utilization=False)
            # A failed initial read leaves baseline unknown, even if later
            # samples succeed. Never substitute a mid-workload reading for it.
            self.baseline_memory = self.peak_memory
            self.started_at = time.perf_counter()
            self.task = asyncio.create_task(self.poll())
        except pynvml.NVMLError as error:
            self.record_error(str(error))
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.stopped.set()
        if self.task is not None:
            await self.task
            self.elapsed_seconds = time.perf_counter() - self.started_at
            self.sample(include_utilization=False)
        if self.initialized:
            try:
                self.nvml.nvmlShutdown()
            except self.nvml.NVMLError as error:
                self.record_error(str(error))

    def snapshot(self):
        return {
            "requested_gpu_uuid": self.requested_uuid,
            "gpu_uuid": self.gpu_uuid,
            "sampling_interval_ms": self.interval_ms,
            "sampling_duration_seconds": self.elapsed_seconds,
            "gpu_utilization_sample_count": self.utilization_samples,
            "average_gpu_utilization_percent": self.utilization_total / self.utilization_samples if self.utilization_samples else None,
            "device_memory_sample_count": self.memory_samples,
            "baseline_device_memory_mib": self.baseline_memory,
            "sampled_peak_device_memory_mib": self.peak_memory,
            "errors": self.errors,
        }


class TorchMemoryMetrics:
    """Tensor allocations in the server process, including loaded model weights.

    This is separate from device VRAM and excludes unused allocator reservations.
    Reset while idle after warmup; read after the engine drains. Errors or a
    non-CUDA test model leave measurements None.
    """
    def __init__(self, device):
        self.device = device
        self.baseline = None
        self.error = None

    def reset(self):
        self.baseline = None
        self.error = None
        if self.device.type != "cuda":
            self.error = "CUDA tensor memory is unavailable for this device"
            return
        import torch
        try:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.baseline = torch.cuda.memory_allocated(self.device) / MIB
        except (RuntimeError, AssertionError) as error:
            self.error = str(error)

    def snapshot(self, synchronize=False):
        peak = None
        if self.baseline is not None:
            import torch
            try:
                if synchronize:
                    torch.cuda.synchronize(self.device)
                peak = torch.cuda.max_memory_allocated(self.device) / MIB
            except (RuntimeError, AssertionError) as error:
                self.error = str(error)
        return {
            "baseline_allocated_gpu_memory_mib": self.baseline,
            "peak_allocated_gpu_memory_mib": peak,
            "gpu_memory_error": self.error,
        }

"""Metric definitions and calculations for GPU inference experiments.

Request durations use milliseconds; experiment durations use seconds. Use
time.perf_counter() for timestamps and only subtract times from the same process.
Exclude startup and warmup from the measured experiment. Its duration starts
before submitting measured jobs and ends after every job completes or fails.

HTTP result dictionaries contain success and latency_ms. Failed requests also
contain error and error_type (rejected, timeout, or error). A successful request
means the client received a 2xx SSE stream and its completion marker. Truncated
streams and in-stream errors are failures, even when the HTTP status is 200.

input_tokens and output_tokens are counts reported by the server. This repo
counts generated EOS; vLLM counts come from its response usage fields. If a
response omits a measurement, missing data stays None (JSON null); we do not
estimate tokens from words or response length. The existing ttft_ms input is a
generation-only diagnostic, from admission to the first generated token, not
client TTFT. It excludes prompt tokenization and stays None for the vLLM adapter.

queue_wait_ms is measured on the server, from entering the FIFO queue until
admission to the active set, including batch collection and waiting for cache
capacity. It excludes network, prompt tokenization, HTTP-framework waiting, and
later waiting for a prefill token budget. Queue
summaries cover successful responses only and stay None if any lacks this field.
Rejected requests never entered the queue and have no queue-wait measurement.
batch_size is the active request count when a request is admitted.
Membership can change on later steps.

client_ttft_ms: HTTP-call start -> first nonempty text chunk received. Empty
outputs have no client TTFT and are excluded from its summary. Role-only and
usage events do not count as text. text_chunk_intervals_ms records gaps between
nonempty text events; these are delivery gaps, not generation TPOT or per-token
latency, because one chunk may contain several tokens. Summaries use successful
requests only, and chunk gaps are pooled across those requests.

generation_tpot_ms: (last_token_ready - first_token_ready) / (output_tokens - 1),
in milliseconds. It includes scheduling between tokens, excludes prefill and
delivery, and is undefined for fewer than two tokens. Average per-request TPOT.

ClientMetrics counts HTTP calls from their actual start until stream completion
or failure, including connection waiting. Scheduled tasks that have not reached
their arrival time are excluded. Metadata requests are excluded too.

GPU definitions and measurements live in gpu_metrics.py: NVML samples device
busyness/VRAM on the client host; server-side PyTorch tracks tensor allocations.
"""

from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class ClientMetrics:
    in_flight: int = 0
    peak_in_flight: int = 0

    def request_started(self):
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)

    def request_finished(self):
        self.in_flight -= 1

    def snapshot(self):
        return {"in_flight_requests": self.in_flight, "peak_in_flight_requests": self.peak_in_flight}


@dataclass
class EngineMetrics:
    """Observed state and forward sizes since the last idle reset.

    Active includes prefill and decode. Reservations count logical tokens, not
    GPU bytes. Forward sizes count requests in actual model invocations,
    including failed invocations, rather than configured batch limits.
    """
    queued_requests: int = 0
    active_requests: int = 0
    reserved_cache_tokens: int = 0
    peak_queued_requests: int = 0
    peak_active_requests: int = 0
    peak_reserved_cache_tokens: int = 0
    prefill_forward_count: int = 0
    prefill_batch_total: int = 0
    peak_prefill_batch_size: int = 0
    decode_forward_count: int = 0
    decode_batch_total: int = 0
    peak_decode_batch_size: int = 0
    profile_enabled: bool = False
    decode_graph_replay_count: int = 0
    profile: dict | None = field(init=False, default=None)

    def __post_init__(self):
        if self.profile_enabled:
            self.profile = {}
            for phase in ("prefill", "decode"):
                self.profile[phase] = {
                    "batch_setup": {"calls": 0, "total_ms": 0.0},
                    "model_forward": {"calls": 0, "total_ms": 0.0},
                    # Keep this stage for comparison with earlier profiles.
                    "cache_save": {"calls": 0, "total_ms": 0.0},
                    "token_handling": {"calls": 0, "total_ms": 0.0},
                }

    def record_stage(self, phase, stage, duration_ms):
        """Accumulate the elapsed time and count of profiled batch stages."""
        timing = self.profile[phase][stage]
        timing["calls"] += 1
        timing["total_ms"] += duration_ms

    def record_queue(self, queued):
        self.queued_requests = queued
        self.peak_queued_requests = max(self.peak_queued_requests, queued)

    def record_active(self, active, reserved):
        self.active_requests = active
        self.reserved_cache_tokens = reserved
        self.peak_active_requests = max(self.peak_active_requests, active)
        self.peak_reserved_cache_tokens = max(self.peak_reserved_cache_tokens, reserved)

    def record_forward(self, phase, size, graph_replay=False):
        """Count real requests in each prefill or decode forward."""
        if phase == "prefill":
            self.prefill_forward_count += 1
            self.prefill_batch_total += size
            self.peak_prefill_batch_size = max(self.peak_prefill_batch_size, size)
        else:
            self.decode_forward_count += 1
            if graph_replay:
                self.decode_graph_replay_count += 1
            self.decode_batch_total += size
            self.peak_decode_batch_size = max(self.peak_decode_batch_size, size)

    def snapshot(self):
        return {
            "queued_requests": self.queued_requests,
            "active_requests": self.active_requests,
            "reserved_cache_tokens": self.reserved_cache_tokens,
            "peak_queued_requests": self.peak_queued_requests,
            "peak_active_requests": self.peak_active_requests,
            "peak_reserved_cache_tokens": self.peak_reserved_cache_tokens,
            "prefill_forward_count": self.prefill_forward_count,
            "mean_prefill_batch_size": self.prefill_batch_total / self.prefill_forward_count if self.prefill_forward_count else None,
            "peak_prefill_batch_size": self.peak_prefill_batch_size,
            "decode_forward_count": self.decode_forward_count,
            "decode_graph_replay_count": self.decode_graph_replay_count,
            "decode_graph_replay_fraction": self.decode_graph_replay_count / self.decode_forward_count if self.decode_forward_count else None,
            "mean_decode_batch_size": self.decode_batch_total / self.decode_forward_count if self.decode_forward_count else None,
            "peak_decode_batch_size": self.peak_decode_batch_size,
            # Take a copy for readers while the worker continues to record.
            "profile": deepcopy(self.profile),
        }


def elapsed_ms(started_at, finished_at):
    """Elapsed milliseconds between two timestamps from the same clock."""
    if finished_at < started_at:
        raise ValueError("finished_at must not be earlier than started_at")
    return (finished_at - started_at) * 1000


def per_second(count, elapsed_seconds):
    """Rate over the whole experiment; undefined for missing counts or zero time."""
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must not be negative")
    if count is None or elapsed_seconds == 0:
        return None
    return count / elapsed_seconds


def percentile(values, percent):
    """A percentile using linear interpolation between sorted observations.

    P50 is the median. When the position lies between two observations, take
    the corresponding fraction of the gap. For [100, 200], P95 is 195.
    """
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    if len(values) == 0:
        return None

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def summarize_http_results(results, elapsed_seconds):
    """Summarize observed HTTP results, including unsuccessful calls in run time.

    requests_per_second: successful responses / experiment seconds.
    output_tokens_per_second: actual tokens in successful responses / seconds.
    average/p50/p95_latency_ms: client HTTP-call latency for successful responses.
    average/p50/p95_queue_wait_ms: server FIFO waiting for successful responses.
    error_rate_percent: all unsuccessful calls / all measured calls * 100.
    Rejections and timeouts also have separate counts so they are distinguishable
    from other errors. Warmup results must not be included in results.

    Token totals/rates are None if any successful response lacks its token count.
    Generation TTFT is averaged only if every successful response reports it.
    With no successes, latency averages/percentiles are None and token total is 0.
    """
    latencies = []
    token_counts = []
    generation_ttfts = []
    generation_tpots = []
    client_ttfts = []
    chunk_intervals = []
    queue_waits = []
    errors = []
    rejected_requests = 0
    timed_out_requests = 0
    other_error_requests = 0

    for result in results:
        if result["success"]:
            latencies.append(result["latency_ms"])
            if result.get("output_tokens") is not None:
                token_counts.append(result["output_tokens"])
            if result.get("ttft_ms") is not None:
                generation_ttfts.append(result["ttft_ms"])
            if result.get("generation_tpot_ms") is not None:
                generation_tpots.append(result["generation_tpot_ms"])
            if result.get("client_ttft_ms") is not None:
                client_ttfts.append(result["client_ttft_ms"])
            chunk_intervals.extend(result.get("text_chunk_intervals_ms", []))
            if result.get("queue_wait_ms") is not None:
                queue_waits.append(result["queue_wait_ms"])
        else:
            errors.append(result["error"])
            error_type = result.get("error_type", "error")
            if error_type == "rejected":
                rejected_requests += 1
            elif error_type == "timeout":
                timed_out_requests += 1
            else:
                other_error_requests += 1

    successful_requests = len(latencies)
    average_latency_ms = None
    average_generation_ttft_ms = None
    if successful_requests > 0:
        average_latency_ms = sum(latencies) / successful_requests
        if len(generation_ttfts) == successful_requests:
            average_generation_ttft_ms = sum(generation_ttfts) / successful_requests

    # A partial token total would make throughput misleading. Every successful
    # request must have a real token count before we report a token rate.
    total_output_tokens = None
    if len(token_counts) == successful_requests:
        total_output_tokens = sum(token_counts)

    # Older reports have no queue time. Avoid summarizing only part of a run.
    if len(queue_waits) != successful_requests:
        queue_waits = []
    average_queue_wait_ms = None
    if len(queue_waits) > 0:
        average_queue_wait_ms = sum(queue_waits) / len(queue_waits)

    error_rate_percent = None
    if len(results) > 0:
        error_rate_percent = len(errors) / len(results) * 100

    return {
        "total_requests": len(results),
        "successful_requests": successful_requests,
        "failed_requests": len(errors),
        "rejected_requests": rejected_requests,
        "timed_out_requests": timed_out_requests,
        "other_error_requests": other_error_requests,
        "error_rate_percent": error_rate_percent,
        # Throughput uses the elapsed time for the whole workload, including drain.
        "requests_per_second": per_second(successful_requests, elapsed_seconds),
        "average_latency_ms": average_latency_ms,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "average_generation_ttft_ms": average_generation_ttft_ms,
        "generation_tpot_samples": len(generation_tpots),
        "average_generation_tpot_ms": sum(generation_tpots) / len(generation_tpots) if generation_tpots else None,
        "p50_generation_tpot_ms": percentile(generation_tpots, 50),
        "p95_generation_tpot_ms": percentile(generation_tpots, 95),
        "requests_with_text": len(client_ttfts),
        "average_client_ttft_ms": sum(client_ttfts) / len(client_ttfts) if client_ttfts else None,
        "p50_client_ttft_ms": percentile(client_ttfts, 50),
        "p95_client_ttft_ms": percentile(client_ttfts, 95),
        "average_text_chunk_interval_ms": sum(chunk_intervals) / len(chunk_intervals) if chunk_intervals else None,
        "p95_text_chunk_interval_ms": percentile(chunk_intervals, 95),
        "average_queue_wait_ms": average_queue_wait_ms,
        "p50_queue_wait_ms": percentile(queue_waits, 50),
        "p95_queue_wait_ms": percentile(queue_waits, 95),
        "elapsed_seconds": elapsed_seconds,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second": per_second(total_output_tokens, elapsed_seconds),
        "errors": errors,
    }

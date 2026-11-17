"""Metric definitions and calculations shared by real and simulated inference.

Request durations use milliseconds; experiment durations use seconds. Use
time.perf_counter() for timestamps and only subtract times from the same process.
Exclude startup and warmup from the measured experiment. Its duration starts
before submitting measured jobs and ends after every job completes or fails.

HTTP result dictionaries contain success and latency_ms. Failed requests also
contain error and error_type (rejected, timeout, or error). A successful request
means the client received a complete 2xx response and decoded its JSON body.

output_tokens and ttft_ms are optional. Real inference reports actual token
counts, including generated EOS. Simulated calls can omit them. Missing data
stays None (JSON null); we do not estimate tokens from words or response length.
The existing ttft_ms input is a generation-only diagnostic, not client TTFT.

queue_wait_ms is measured on the server, from entering the FIFO queue until batch
execution begins, including batch collection. It excludes network and HTTP-framework
waiting. Queue
summaries cover successful responses only and stay None if any lacks this field.
Rejected requests never entered the queue and have no queue-wait measurement.
batch_size is the number of requests at dispatch, not a per-token active count.
"""


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
        "requests_per_second": per_second(successful_requests, elapsed_seconds),
        "average_latency_ms": average_latency_ms,
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "average_generation_ttft_ms": average_generation_ttft_ms,
        "average_queue_wait_ms": average_queue_wait_ms,
        "p50_queue_wait_ms": percentile(queue_waits, 50),
        "p95_queue_wait_ms": percentile(queue_waits, 95),
        "elapsed_seconds": elapsed_seconds,
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second": per_second(total_output_tokens, elapsed_seconds),
        "errors": errors,
    }


# FUTURE: observed request counts
# client_in_flight: HTTP calls started but not yet completed or failed.
# queued_requests: accepted jobs waiting for admission to the inference engine.
# active_requests: admitted jobs that have not finished, including time between
# their scheduled token steps. An active request is not necessarily on the GPU.
# Record these counts as they change, then report their peaks. The benchmark's
# configured concurrency limit is not an observation of any of these counts.
# Needed: client start/finish events and the engine's request lifecycle.

# FUTURE: client_ttft_ms
# Client HTTP-call start -> first nonempty response text received over /stream.
# Needed: a streaming benchmark that records the first text arrival. /generate
# returns a full answer, so its response's ttft_ms cannot provide this measurement.
# Text chunks can contain multiple tokens; first text arrival is not necessarily
# arrival of exactly one token. Requests with no visible text have no client TTFT.

# FUTURE: generation_tpot_ms
# (last_token_ready - first_token_ready) * 1000 / (output_tokens - 1).
# Use server timestamps when selected token IDs are ready, including scheduling
# delays between tokens. Exclude initial queueing, prefill, and text delivery.
# Needed: reliable first/last-token timestamps. Current latency_ms sometimes
# includes an unused final forward pass, so do not use it to calculate TPOT.
# Keep None for fewer than two output tokens. Revisit timing with batching too.

# FUTURE: GPU utilization
# GPU busyness: sampled percentage of time during which GPU kernels execute.
# Needed: a GPU sampler and defined experiment start/stop boundaries. Record the
# sampling interval and average samples over the run. This is not a measure of
# achieved FLOPS or compute efficiency. Simulated inference should report None.

# FUTURE: peak_allocated_gpu_memory_mib
# Peak PyTorch tensor allocation on the selected GPU during the experiment.
# Needed: server-side reset_peak_memory_stats() after warmup and a final reading
# of max_memory_allocated() after the experiment's server work finishes.
# Divide bytes by 1024 * 1024. Include the loaded model and record its baseline.
# This excludes allocator-reserved unused memory and other non-tensor GPU usage;
# it is not total device VRAM usage. Simulated inference should report None.

# FUTURE: actual_decode_batch_size
# Number of requests advanced together in one model decode forward pass.
# Needed: a batched decode loop recording the size of each forward's batch.
# Report mean and peak over decode forwards. Keep prefill measurements separate.
# Several active requests taking separate turns still have a decode batch size
# of one. Never infer actual batch size from configured client concurrency.

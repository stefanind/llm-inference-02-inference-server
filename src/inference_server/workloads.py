"""Build W1-W6 as scheduled requests. Nothing here sends HTTP calls or runs a model.

arrival_ms is the intended client-call start, relative to an experiment's start.
It does not guarantee the order requests reach the server. The runner sends
prompt and max_new_tokens; IDs and arrival times remain client metadata.
"""

import math
import random
from dataclasses import dataclass


W6_RATES = [1, 2, 5, 10, 20]


@dataclass
class ScheduledRequest:
    request_id: str
    user_id: int | None
    arrival_ms: float
    prompt: str
    # An output limit, not a promise that the model will generate this many tokens.
    max_new_tokens: int = 32


def make_w1() -> list[ScheduledRequest]:
    """W1: one request at time zero, for baseline execution."""
    return [ScheduledRequest(
        request_id="W1-u1-r1",
        user_id=1,
        arrival_ms=0,
        prompt="Request W1-u1-r1",
    )]


def make_w2() -> list[ScheduledRequest]:
    """W2: 20 users submit one request each at time zero."""
    requests = []
    for user_id in range(1, 21):
        request_id = f"W2-u{user_id}-r1"
        requests.append(ScheduledRequest(
            request_id=request_id,
            user_id=user_id,
            arrival_ms=0,
            prompt=f"Request {request_id}",
        ))
    return requests


def make_w3(seed: int = 42) -> list[ScheduledRequest]:
    """W3: 20 users submit once each, randomly within a 10-second window."""
    # A local random generator makes this repeatable without changing global randomness.
    generator = random.Random(seed)
    requests = []
    for user_id in range(1, 21):
        request_id = f"W3-u{user_id}-r1"
        requests.append(ScheduledRequest(
            request_id=request_id,
            user_id=user_id,
            arrival_ms=generator.randint(0, 10000),
            prompt=f"Request {request_id}",
        ))

    # Chronological order makes the schedule easier to inspect.
    requests.sort(key=lambda request: request.arrival_ms)
    return requests


def make_w4(short_prompt: str, long_prompt: str) -> list[ScheduledRequest]:
    """W4: five requests for each short/long prompt and output-limit combination.

    Supply prepared prompt texts; token lengths must be checked separately with
    the real model's tokenizer. We do not modify them or guess tokens from words.
    Simulation still uses its fixed delay for every request.
    """
    groups = [
        (short_prompt, 32),
        (short_prompt, 256),
        (long_prompt, 32),
        (long_prompt, 256),
    ]
    requests = []
    user_id = 1
    for prompt, max_new_tokens in groups:
        for _ in range(5):
            requests.append(ScheduledRequest(
                request_id=f"W4-u{user_id}-r1",
                user_id=user_id,
                arrival_ms=0,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            ))
            user_id += 1
    return requests


def make_w5() -> list[ScheduledRequest]:
    """W5: 20 users each submit 20 requests at time zero: 400 total.

    User IDs label sources; they do not impose a per-user concurrency limit.
    Once admission control exists, explicit rejections are valid outcomes.
    """
    requests = []
    for user_id in range(1, 21):
        for request_number in range(1, 21):
            request_id = f"W5-u{user_id}-r{request_number}"
            requests.append(ScheduledRequest(
                request_id=request_id,
                user_id=user_id,
                arrival_ms=0,
                prompt=f"Request {request_id}",
            ))
    return requests


def make_w6(requests_per_second: float, duration_seconds: float = 30) -> list[ScheduledRequest]:
    """W6: one fixed-rate case. Run each rate in W6_RATES separately.

    Schedule the first request at zero, then at fixed intervals strictly before
    the end of the arrival window. Earlier responses do not affect these times.
    Run rates separately after earlier server work has finished. The runner waits
    for all client calls; a timeout cannot confirm that server work has stopped.
    """
    if not math.isfinite(requests_per_second) or requests_per_second <= 0:
        raise ValueError("requests_per_second must be positive and finite")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")

    requests = []
    request_number = 0
    while True:
        # Calculate from the index so rounding does not accumulate across intervals.
        arrival_ms = request_number / requests_per_second * 1000
        if arrival_ms >= duration_seconds * 1000:
            break
        request_id = f"W6-{requests_per_second}rps-r{request_number + 1}"
        requests.append(ScheduledRequest(
            request_id=request_id,
            user_id=None,
            arrival_ms=arrival_ms,
            prompt=f"Request {request_id}",
        ))
        request_number += 1
    return requests

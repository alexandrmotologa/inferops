"""Synthetic load generator and latency profiler for inference endpoints."""

import asyncio
import time
from dataclasses import dataclass
from typing import List

import httpx


@dataclass
class BenchmarkResult:
    model_name: str
    num_requests: int
    concurrency: int
    successful_requests: int
    failed_requests: int
    total_tokens_generated: int
    elapsed_time_sec: float
    tokens_per_second: float
    avg_ttft_ms: float
    p95_ttft_ms: float
    min_ttft_ms: float
    max_ttft_ms: float


async def _run_benchmark_request(
    client: httpx.AsyncClient,
    url: str,
    model_alias: str,
    prompt: str,
    max_tokens: int,
) -> tuple[bool, float, int]:
    """Execute a single streaming request and record TTFT and token count."""
    payload = {
        "model": model_alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    start = time.perf_counter()
    first_token_latency = 0.0
    tokens = 0

    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                return False, 0.0, 0

            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if first_token_latency == 0.0:
                        first_token_latency = (time.perf_counter() - start) * 1000.0
                    tokens += 1
            return True, first_token_latency, tokens
    except Exception:
        return False, 0.0, 0


async def run_model_benchmark(
    host: str,
    port: int,
    model_alias: str,
    num_requests: int = 10,
    concurrency: int = 2,
    max_tokens: int = 128,
    prompt: str = "Explain the difference between TCP and UDP in 2 sentences.",
) -> BenchmarkResult:
    """Run concurrent benchmark requests and compute performance metrics."""
    target_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{target_host}:{port}/v1/chat/completions"

    ttft_list: List[float] = []
    total_tokens = 0
    success_count = 0
    failure_count = 0

    semaphore = asyncio.Semaphore(concurrency)

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal total_tokens, success_count, failure_count
        async with semaphore:
            success, ttft, tokens = await _run_benchmark_request(
                client=client,
                url=url,
                model_alias=model_alias,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            if success:
                success_count += 1
                total_tokens += tokens
                if ttft > 0:
                    ttft_list.append(ttft)
            else:
                failure_count += 1

    overall_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [worker(client) for _ in range(num_requests)]
        await asyncio.gather(*tasks)

    overall_time = max(0.001, time.perf_counter() - overall_start)
    tok_per_sec = total_tokens / overall_time

    if ttft_list:
        ttft_list.sort()
        avg_ttft = sum(ttft_list) / len(ttft_list)
        p95_idx = int(len(ttft_list) * 0.95)
        p95_ttft = ttft_list[min(p95_idx, len(ttft_list) - 1)]
        min_ttft = ttft_list[0]
        max_ttft = ttft_list[-1]
    else:
        avg_ttft = p95_ttft = min_ttft = max_ttft = 0.0

    return BenchmarkResult(
        model_name=model_alias,
        num_requests=num_requests,
        concurrency=concurrency,
        successful_requests=success_count,
        failed_requests=failure_count,
        total_tokens_generated=total_tokens,
        elapsed_time_sec=round(overall_time, 2),
        tokens_per_second=round(tok_per_sec, 1),
        avg_ttft_ms=round(avg_ttft, 1),
        p95_ttft_ms=round(p95_ttft, 1),
        min_ttft_ms=round(min_ttft, 1),
        max_ttft_ms=round(max_ttft, 1),
    )

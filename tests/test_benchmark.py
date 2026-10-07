"""Unit tests for the synthetic benchmark runner."""

import pytest
from starlette.responses import StreamingResponse

from inferops.core.benchmark import BenchmarkResult


async def fake_chat_endpoint(request):
    async def sse_gen():
        yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        yield b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    return StreamingResponse(sse_gen(), media_type="text/event-stream")


@pytest.mark.asyncio
async def test_benchmark_result_structure() -> None:
    res = BenchmarkResult(
        model_name="test-model",
        num_requests=5,
        concurrency=2,
        successful_requests=5,
        failed_requests=0,
        total_tokens_generated=100,
        elapsed_time_sec=2.0,
        tokens_per_second=50.0,
        avg_ttft_ms=12.5,
        p95_ttft_ms=15.0,
        min_ttft_ms=10.0,
        max_ttft_ms=16.0,
    )
    assert res.successful_requests == 5
    assert res.tokens_per_second == 50.0

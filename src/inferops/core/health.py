"""HTTP health and readiness probes with timeout and backoff."""

import asyncio
import time
from typing import Callable, Optional
import httpx


async def check_http_health(
    host: str,
    port: int,
    endpoint: str = "/health",
    timeout: float = 1.5,
) -> bool:
    """Send a fast GET request to verify readiness. Returns True if status 200."""
    target_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{target_host}:{port}{endpoint}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


async def wait_for_health(
    host: str,
    port: int,
    endpoint: str = "/health",
    timeout_seconds: float = 180.0,
    poll_interval: float = 1.5,
    is_alive_check: Optional[Callable[[], bool]] = None,
) -> bool:
    """Poll endpoint until service returns HTTP 200 or timeout is reached."""
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        # Check if underlying process died early
        if is_alive_check and not is_alive_check():
            return False

        if await check_http_health(host, port, endpoint, timeout=poll_interval):
            return True

        await asyncio.sleep(poll_interval)

    return False

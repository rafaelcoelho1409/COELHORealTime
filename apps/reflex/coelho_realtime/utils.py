import httpx

# =============================================================================
# HTTP Client with Connection Pooling
# =============================================================================
# Reuse connections to reduce latency on repeated requests.
# Default timeout reduced to 10s for faster failure detection.
# =============================================================================

# Shared client instance for connection reuse
_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Get or create shared HTTP client with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _http_client


async def httpx_client_get(timeout: float = 10.0, **kwargs):
    """GET request with configurable timeout using shared client."""
    client = get_http_client()
    return await client.get(timeout=timeout, **kwargs)


async def httpx_client_post(timeout: float = 10.0, **kwargs):
    """POST request with configurable timeout using shared client."""
    client = get_http_client()
    return await client.post(timeout=timeout, **kwargs)
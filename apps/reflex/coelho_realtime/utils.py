import httpx

# Default timeout for all HTTP requests (30 seconds)
DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


## FUNCTIONS
async def httpx_client_get(timeout: float = 30.0, **kwargs):
    """GET request with configurable timeout."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.get(**kwargs)


async def httpx_client_post(timeout: float = 30.0, **kwargs):
    """POST request with configurable timeout."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.post(**kwargs)
"""
YellowBrick Visualization Cache using Redis.

Caches generated PNG visualizations to avoid redundant computation.
Cache is invalidated when new training completes.

Key structure:
    yellowbrick:{project_name}:{visualizer_name} → PNG bytes

TTL: 1 hour (configurable) - auto-cleanup for stale visualizations
"""
import os
import redis
from typing import Optional
from urllib.parse import urlparse


def parse_redis_url(url: str) -> tuple[str, int]:
    """Parse Redis URL to extract host and port.

    Handles formats:
    - redis://hostname:port
    - hostname:port
    - hostname
    """
    if url.startswith("redis://"):
        parsed = urlparse(url)
        return parsed.hostname or "localhost", parsed.port or 6379
    elif ":" in url:
        host, port = url.rsplit(":", 1)
        return host, int(port)
    else:
        return url, 6379


# Redis configuration from environment (set in configmap)
# REDIS_HOST can be URL format (redis://host:port) or just hostname
_redis_url = os.getenv("REDIS_HOST", "coelho-realtime-redis-master")
REDIS_HOST, REDIS_PORT = parse_redis_url(_redis_url)
REDIS_DB = int(os.getenv("REDIS_DB", "1"))  # Use DB 1 for sklearn (DB 0 for Reflex)

# Cache TTL in seconds (1 hour default)
CACHE_TTL_SECONDS = int(os.getenv("YELLOWBRICK_CACHE_TTL", "3600"))

# Key prefix
KEY_PREFIX = "yellowbrick"


class VisualizationCache:
    """
    Redis-based cache for YellowBrick visualizations.

    Provides:
    - get(): Retrieve cached visualization
    - set(): Store visualization with TTL
    - invalidate(): Clear all visualizations for a project (on new training)
    - clear_all(): Clear entire cache
    """

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._connected = False

    @property
    def client(self) -> Optional[redis.Redis]:
        """Lazy connection to Redis."""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    decode_responses=False,  # We store binary PNG data
                    socket_timeout=5,
                    socket_connect_timeout=5,
                )
                # Test connection
                self._client.ping()
                self._connected = True
                print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT} (db={REDIS_DB})")
            except redis.ConnectionError as e:
                print(f"Warning: Could not connect to Redis: {e}")
                print("Visualization caching disabled - will generate on each request")
                self._client = None
                self._connected = False
        return self._client

    def _make_key(self, project_name: str, visualizer_name: str) -> str:
        """Create Redis key for a visualization."""
        # Normalize project name (replace spaces with underscores)
        project_key = project_name.lower().replace(" ", "_")
        return f"{KEY_PREFIX}:{project_key}:{visualizer_name}"

    def get(self, project_name: str, visualizer_name: str) -> Optional[bytes]:
        """
        Get cached visualization.

        Args:
            project_name: Project name (e.g., "Transaction Fraud Detection")
            visualizer_name: Visualizer name (e.g., "PCA", "Rank1D")

        Returns:
            PNG bytes if cached, None otherwise
        """
        if not self.client:
            return None

        key = self._make_key(project_name, visualizer_name)
        try:
            data = self.client.get(key)
            if data:
                print(f"Cache HIT: {key}")
                return data
            print(f"Cache MISS: {key}")
            return None
        except redis.RedisError as e:
            print(f"Redis error on get: {e}")
            return None

    def set(
        self,
        project_name: str,
        visualizer_name: str,
        image_bytes: bytes,
        ttl: int = None
    ) -> bool:
        """
        Store visualization in cache.

        Args:
            project_name: Project name
            visualizer_name: Visualizer name
            image_bytes: PNG image data
            ttl: Time-to-live in seconds (default: CACHE_TTL_SECONDS)

        Returns:
            True if stored successfully, False otherwise
        """
        if not self.client:
            return False

        key = self._make_key(project_name, visualizer_name)
        ttl = ttl or CACHE_TTL_SECONDS

        try:
            self.client.setex(key, ttl, image_bytes)
            print(f"Cache SET: {key} (TTL: {ttl}s, size: {len(image_bytes)} bytes)")
            return True
        except redis.RedisError as e:
            print(f"Redis error on set: {e}")
            return False

    def invalidate(self, project_name: str) -> int:
        """
        Invalidate all cached visualizations for a project.

        Call this when new training completes to ensure fresh visualizations.

        Args:
            project_name: Project name

        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0

        project_key = project_name.lower().replace(" ", "_")
        pattern = f"{KEY_PREFIX}:{project_key}:*"

        try:
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                deleted = self.client.delete(*keys)
                print(f"Cache INVALIDATED: {deleted} visualizations for '{project_name}'")
                return deleted
            return 0
        except redis.RedisError as e:
            print(f"Redis error on invalidate: {e}")
            return 0

    def clear_all(self) -> int:
        """
        Clear all YellowBrick visualizations from cache.

        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0

        pattern = f"{KEY_PREFIX}:*"

        try:
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                deleted = self.client.delete(*keys)
                print(f"Cache CLEARED: {deleted} visualizations")
                return deleted
            return 0
        except redis.RedisError as e:
            print(f"Redis error on clear_all: {e}")
            return 0

    def status(self) -> dict:
        """
        Get cache status information.

        Returns:
            Dict with connection status and cached keys count
        """
        if not self.client:
            return {
                "connected": False,
                "message": "Redis not connected - caching disabled",
            }

        try:
            pattern = f"{KEY_PREFIX}:*"
            keys = list(self.client.scan_iter(match=pattern))

            # Group by project
            projects = {}
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) >= 3:
                    project = parts[1]
                    visualizer = parts[2]
                    if project not in projects:
                        projects[project] = []
                    projects[project].append(visualizer)

            return {
                "connected": True,
                "redis_host": REDIS_HOST,
                "redis_port": REDIS_PORT,
                "redis_db": REDIS_DB,
                "ttl_seconds": CACHE_TTL_SECONDS,
                "cached_count": len(keys),
                "cached_by_project": projects,
            }
        except redis.RedisError as e:
            return {
                "connected": False,
                "error": str(e),
            }


# Global singleton instance
visualization_cache = VisualizationCache()

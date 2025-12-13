import reflex as rx
import os

# Note: All config values can be overridden via environment variables with REFLEX_ prefix
config = rx.Config(
    app_name = "coelho_realtime",
    # Standard Reflex ports (override with REFLEX_FRONTEND_PORT / REFLEX_BACKEND_PORT)
    frontend_port = int(os.getenv("REFLEX_FRONTEND_PORT", "3000")),
    backend_port = int(os.getenv("REFLEX_BACKEND_PORT", "8000")),
    backend_host = os.getenv("REFLEX_BACKEND_HOST", "0.0.0.0"),
    # API URL: Where browser connects to backend
    # Standalone Docker: http://localhost:8000
    # K8s: set via REFLEX_API_URL env var
    api_url = os.getenv("REFLEX_API_URL", "http://localhost:8000"),
    # Deploy URL: Where frontend is accessible
    deploy_url = os.getenv("REFLEX_DEPLOY_URL", "http://localhost:3000"),
    # Environment mode
    env = rx.Env.PROD if os.getenv("REFLEX_ENV") == "PROD" else rx.Env.DEV,
    # Redis URL
    # Standalone Docker: redis://localhost (Redis in same container)
    # K8s: set via REFLEX_REDIS_URL env var to point to Redis service
    redis_url = os.getenv("REFLEX_REDIS_URL", "redis://localhost"),
    # CORS settings
    cors_allowed_origins = os.getenv("REFLEX_CORS_ALLOWED_ORIGINS", "*").split(","),
)

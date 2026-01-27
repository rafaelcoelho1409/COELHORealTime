"""
Lightweight Prometheus metrics HTTP server for Kafka Producers.

Uses MultiProcessCollector to aggregate metrics from all producer processes.
Serves on port 8000 at /metrics endpoint.
"""
import os
import sys
from prometheus_client import (
    CollectorRegistry,
    multiprocess,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from http.server import HTTPServer, BaseHTTPRequestHandler


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves Prometheus metrics."""

    def do_GET(self):
        if self.path == "/metrics":
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            output = generate_latest(registry)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(output)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default request logging to reduce noise."""
        pass


def main():
    port = int(os.environ.get("METRICS_PORT", "8000"))
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"Prometheus metrics server listening on port {port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

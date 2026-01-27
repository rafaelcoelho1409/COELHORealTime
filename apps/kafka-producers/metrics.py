"""
Prometheus Metrics Registry - Kafka Producers

Shared metrics module using multiprocess mode for multiple producer processes.
All metrics use multiprocess-compatible modes where needed.
"""
from prometheus_client import Counter, Histogram, Gauge


# =============================================================================
# Message Metrics
# =============================================================================
MESSAGES_SENT_TOTAL = Counter(
    "kafka_producer_messages_sent_total",
    "Total messages sent to Kafka",
    ["topic", "producer"],
)

ERRORS_TOTAL = Counter(
    "kafka_producer_errors_total",
    "Total producer errors",
    ["topic", "producer", "error_type"],
)

SEND_DURATION_SECONDS = Histogram(
    "kafka_producer_send_duration_seconds",
    "Kafka send latency in seconds",
    ["topic"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

MESSAGE_SIZE_BYTES = Histogram(
    "kafka_producer_message_size_bytes",
    "Message size in bytes",
    ["topic"],
    buckets=[100, 500, 1000, 5000, 10000, 50000],
)

# =============================================================================
# Connection Metrics
# =============================================================================
CONNECTED = Gauge(
    "kafka_producer_connected",
    "Whether producer is connected (1=yes, 0=no)",
    ["producer"],
    multiprocess_mode="livesum",
)

CONNECTION_RETRIES_TOTAL = Counter(
    "kafka_producer_connection_retries_total",
    "Total connection retries",
    ["producer"],
)

# =============================================================================
# Operational Metrics
# =============================================================================
LAST_MESSAGE_TIMESTAMP = Gauge(
    "kafka_producer_last_message_timestamp",
    "Unix timestamp of last message sent",
    ["producer"],
    multiprocess_mode="max",
)

FRAUD_RATIO = Gauge(
    "kafka_producer_fraud_ratio",
    "Current fraud ratio in TFD producer",
    multiprocess_mode="livesum",
)

ACTIVE_SESSIONS = Gauge(
    "kafka_producer_active_sessions",
    "Number of active ECCI sessions",
    multiprocess_mode="livesum",
)

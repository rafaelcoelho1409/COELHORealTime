#!/bin/bash
# Test Alerting Rules - Validates all PromQL queries against Prometheus
#
# Usage: ./scripts/test-alerting-rules.sh
#
# Prerequisites:
#   - kubectl configured for your cluster
#   - Prometheus running in coelho-realtime namespace

set -e

PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
NAMESPACE="coelho-realtime"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Alerting Rules Validation Script"
echo "=========================================="
echo ""

# Check if port-forward is needed
if ! curl -s "${PROMETHEUS_URL}/api/v1/status/config" > /dev/null 2>&1; then
    echo -e "${YELLOW}Prometheus not accessible at ${PROMETHEUS_URL}${NC}"
    echo "Starting port-forward..."
    kubectl port-forward -n ${NAMESPACE} svc/coelho-realtime-kube-prome-prometheus 9090:9090 &
    PF_PID=$!
    sleep 3
    trap "kill $PF_PID 2>/dev/null" EXIT
fi

# Function to test a PromQL query
test_query() {
    local name="$1"
    local query="$2"
    local description="$3"

    # URL encode the query
    encoded_query=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$query'''))")

    response=$(curl -s "${PROMETHEUS_URL}/api/v1/query?query=${encoded_query}")
    status=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'error'))" 2>/dev/null || echo "error")

    if [ "$status" = "success" ]; then
        # Check if there are results
        result_count=$(echo "$response" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('data', {}).get('result', [])))" 2>/dev/null || echo "0")
        echo -e "${GREEN}✓${NC} ${name}"
        echo "  Query: ${query:0:80}..."
        echo "  Status: success (${result_count} results)"
    else
        error_msg=$(echo "$response" | python3 -c "import sys, json; print(json.load(sys.stdin).get('error', 'Unknown error'))" 2>/dev/null || echo "Parse error")
        echo -e "${RED}✗${NC} ${name}"
        echo "  Query: ${query:0:80}..."
        echo "  Error: ${error_msg}"
        return 1
    fi
    echo ""
}

echo "Testing River ML Service Alerts..."
echo "-----------------------------------"
test_query "RiverMLDown" 'up{job="river"} == 0' "River service down"
test_query "RiverTrainingStalled" 'rate(river_samples_processed_total[10m]) == 0' "No samples processed"
test_query "RiverHighPredictionLatency" 'histogram_quantile(0.99, rate(river_prediction_duration_seconds_bucket[5m])) > 0.5' "High prediction latency"

echo "Testing Sklearn Service Alerts..."
echo "----------------------------------"
test_query "SklearnDown" 'up{job="sklearn"} == 0' "Sklearn service down"
test_query "SklearnHighPredictionLatency" 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{job="sklearn", handler="/predict"}[5m])) > 2' "High prediction latency"
test_query "SklearnHighErrorRate" 'rate(http_requests_total{job="sklearn", status=~"5.."}[5m]) / rate(http_requests_total{job="sklearn"}[5m]) > 0.05' "High error rate"

echo "Testing Kafka Alerts..."
echo "------------------------"
test_query "KafkaDown" 'up{job=~".*kafka.*"} == 0' "Kafka broker down"
test_query "KafkaProducersDown" 'kube_deployment_status_replicas_available{deployment="coelho-realtime-kafka-producers", namespace="coelho-realtime"} == 0' "Kafka producers down"
test_query "KafkaProducersCrashLooping" 'rate(kube_pod_container_status_restarts_total{pod=~"coelho-realtime-kafka-producers.*", namespace="coelho-realtime"}[15m]) > 0' "Kafka producers crash looping"

echo "Testing MLflow Alerts..."
echo "-------------------------"
test_query "MLflowDown" 'up{job="mlflow"} == 0' "MLflow service down"

echo "Testing PostgreSQL Alerts..."
echo "-----------------------------"
test_query "PostgreSQLDown" 'pg_up == 0' "PostgreSQL down"
test_query "PostgreSQLTooManyConnections" 'sum by (instance) (pg_stat_activity_count) > (pg_settings_max_connections * 0.8)' "Too many connections"

echo "Testing Redis Alerts..."
echo "------------------------"
test_query "RedisDown" 'redis_up == 0' "Redis down"
test_query "RedisHighMemoryUsage" '(redis_memory_used_bytes / redis_memory_max_bytes) * 100 > 90' "High memory usage"

echo "Testing MinIO Alerts..."
echo "------------------------"
test_query "MinIODown" 'up{job="minio"} == 0' "MinIO down"
test_query "MinIODiskOffline" 'minio_cluster_disk_offline_total > 0' "Disk offline"

echo "Testing Reflex Alerts..."
echo "-------------------------"
test_query "ReflexDown" 'up{job="reflex"} == 0' "Reflex down"
test_query "ReflexHighLatency" 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket{job="reflex"}[5m])) > 3' "High latency"

echo "Testing Spark Alerts..."
echo "------------------------"
test_query "SparkMasterDown" 'up{job=~".*spark-master.*"} == 0' "Spark Master down"
test_query "SparkWorkerDown" 'up{job=~".*spark-worker.*"} == 0' "Spark Worker down"
test_query "SparkStreamingDown" 'kube_deployment_status_replicas_available{deployment="coelho-realtime-spark-streaming", namespace="coelho-realtime"} == 0' "Spark Streaming down"

echo "Testing General Alerts..."
echo "--------------------------"
test_query "HighMemoryUsage" '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85' "High memory usage"
test_query "HighCpuUsage" '100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85' "High CPU usage"
test_query "PodCrashLooping" 'rate(kube_pod_container_status_restarts_total{namespace="coelho-realtime"}[15m]) > 0' "Pod crash looping"

echo ""
echo "=========================================="
echo "  Validation Complete"
echo "=========================================="

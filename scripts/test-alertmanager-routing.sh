#!/bin/bash
# Test Alertmanager Routing - Validates alert routing configuration
#
# Usage: ./scripts/test-alertmanager-routing.sh
#
# Prerequisites:
#   - kubectl configured for your cluster
#   - Alertmanager running in coelho-realtime namespace

set -e

ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://localhost:9094}"
NAMESPACE="coelho-realtime"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Alertmanager Routing Test Script"
echo "=========================================="
echo ""

# Check if port-forward is needed
if ! curl -s "${ALERTMANAGER_URL}/api/v2/status" > /dev/null 2>&1; then
    echo -e "${YELLOW}Alertmanager not accessible at ${ALERTMANAGER_URL}${NC}"
    echo "Starting port-forward..."
    kubectl port-forward -n ${NAMESPACE} svc/coelho-realtime-kube-prome-alertmanager 9094:9094 &
    PF_PID=$!
    sleep 3
    trap "kill $PF_PID 2>/dev/null" EXIT
fi

echo -e "${CYAN}1. Checking Alertmanager Status${NC}"
echo "--------------------------------"
status=$(curl -s "${ALERTMANAGER_URL}/api/v2/status")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Alertmanager is running${NC}"
    echo "$status" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Cluster: {data.get('cluster', {}).get('status', 'N/A')}\")
print(f\"  Uptime: {data.get('uptime', 'N/A')}\")
"
else
    echo -e "${RED}✗ Cannot connect to Alertmanager${NC}"
    exit 1
fi
echo ""

echo -e "${CYAN}2. Current Active Alerts${NC}"
echo "-------------------------"
alerts=$(curl -s "${ALERTMANAGER_URL}/api/v2/alerts")
alert_count=$(echo "$alerts" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
echo "Active alerts: ${alert_count}"

if [ "$alert_count" != "0" ]; then
    echo "$alerts" | python3 -c "
import sys, json
alerts = json.load(sys.stdin)
for alert in alerts[:10]:  # Show first 10
    labels = alert.get('labels', {})
    print(f\"  - {labels.get('alertname', 'N/A')} [{labels.get('severity', 'N/A')}] - {labels.get('namespace', 'N/A')}\")
if len(alerts) > 10:
    print(f\"  ... and {len(alerts) - 10} more\")
"
fi
echo ""

echo -e "${CYAN}3. Current Silences${NC}"
echo "--------------------"
silences=$(curl -s "${ALERTMANAGER_URL}/api/v2/silences")
silence_count=$(echo "$silences" | python3 -c "
import sys, json
silences = json.load(sys.stdin)
active = [s for s in silences if s.get('status', {}).get('state') == 'active']
print(len(active))
" 2>/dev/null || echo "0")
echo "Active silences: ${silence_count}"
echo ""

echo -e "${CYAN}4. Testing Alert Routing${NC}"
echo "-------------------------"
echo "Simulating alerts to test routing configuration..."
echo ""

# Test routing for different severity levels
test_routing() {
    local alertname="$1"
    local severity="$2"
    local namespace="$3"

    # Create test alert JSON
    local alert_json=$(cat <<EOF
[{
  "labels": {
    "alertname": "${alertname}",
    "severity": "${severity}",
    "namespace": "${namespace}",
    "instance": "test:9090"
  },
  "annotations": {
    "summary": "Test alert for routing verification",
    "description": "This is a test alert"
  },
  "startsAt": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "generatorURL": "http://prometheus:9090/test"
}]
EOF
)

    echo "  Testing: ${alertname} (severity=${severity})"

    # Send test alert (dry-run by checking the receiver)
    response=$(curl -s -X POST "${ALERTMANAGER_URL}/api/v2/alerts" \
        -H "Content-Type: application/json" \
        -d "${alert_json}" 2>&1)

    if [ $? -eq 0 ]; then
        echo -e "    ${GREEN}✓ Alert accepted${NC}"
    else
        echo -e "    ${RED}✗ Alert rejected: ${response}${NC}"
    fi
}

# Test different alert scenarios
test_routing "TestCriticalAlert" "critical" "coelho-realtime"
test_routing "TestWarningAlert" "warning" "coelho-realtime"
test_routing "TestInfoAlert" "info" "coelho-realtime"
test_routing "Watchdog" "none" "coelho-realtime"

echo ""
echo -e "${CYAN}5. Alertmanager Configuration${NC}"
echo "-------------------------------"
echo "Fetching current configuration..."

config=$(curl -s "${ALERTMANAGER_URL}/api/v2/status" | python3 -c "
import sys, json
data = json.load(sys.stdin)
config = data.get('config', {}).get('original', 'N/A')
# Print first 50 lines
lines = config.split('\n')[:50]
for line in lines:
    print(line)
if len(config.split('\n')) > 50:
    print('... (truncated)')
" 2>/dev/null || echo "Could not fetch config")

echo "$config"
echo ""

echo -e "${CYAN}6. Receivers Summary${NC}"
echo "---------------------"
curl -s "${ALERTMANAGER_URL}/api/v2/receivers" | python3 -c "
import sys, json
receivers = json.load(sys.stdin)
print(f'Configured receivers: {len(receivers)}')
for r in receivers:
    print(f\"  - {r.get('name', 'N/A')}\")
" 2>/dev/null || echo "Could not fetch receivers"

echo ""
echo "=========================================="
echo "  Routing Test Complete"
echo "=========================================="
echo ""
echo "Note: Test alerts were sent to Alertmanager."
echo "They will be automatically resolved after the default timeout."
echo ""
echo "To manually expire test alerts:"
echo "  curl -X POST ${ALERTMANAGER_URL}/api/v2/alerts -d '[{\"labels\":{\"alertname\":\"TestCriticalAlert\"},\"endsAt\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}]'"

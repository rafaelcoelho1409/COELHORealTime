#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ICONS_DIR="$SCRIPT_DIR/icons"
mkdir -p "$ICONS_DIR"

# ── Icon downloads ──────────────────────────────────────────
# Uses devicon CDN (jsdelivr) for standard tech icons as SVGs.
# Local icons already in diagrams/ are referenced directly.
declare -A ICONS=(
  [terraform]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/terraform/terraform-original.svg"
  [docker]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/docker/docker-original.svg"
  [gitlab]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/gitlab/gitlab-original.svg"
  [helm]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/helm/helm-original.svg"
  [redis]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/redis/redis-original.svg"
  [python]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg"
  [prometheus]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/prometheus/prometheus-original.svg"
  [grafana]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/grafana/grafana-original.svg"
  [postgresql]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg"
  [kafka]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/apachekafka/apachekafka-original.svg"
  [argocd]="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/argocd/argocd-original.svg"
)

echo "Downloading icons..."
for name in "${!ICONS[@]}"; do
  dest="$ICONS_DIR/${name}.svg"
  if [[ -f "$dest" ]]; then
    echo "  --  ${name}.svg (cached)"
  else
    if curl -sfL -o "$dest" "${ICONS[$name]}"; then
      echo "  OK  ${name}.svg"
    else
      echo "  FAIL ${name}.svg"
    fi
  fi
done

# ── Generate diagram ────────────────────────────────────────
echo ""
echo "Generating D2 diagram..."

d2 --layout=elk \
   --scale=0.6 \
   --pad=40 \
   coelho_realtime_architecture.d2 \
   coelho_realtime_architecture.svg 2>&1

echo ""
echo "Diagram generated: coelho_realtime_architecture.svg"

# ── Convert SVG to PNG ──────────────────────────────────────
echo "Converting SVG to PNG..."
rsvg-convert -d 300 -p 300 \
  coelho_realtime_architecture.svg \
  -o coelho_realtime_architecture.png 2>&1

echo "Diagram converted: coelho_realtime_architecture.png"

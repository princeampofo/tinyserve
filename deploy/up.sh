#!/usr/bin/env bash
# One command: tinyserve on a local Kubernetes cluster (kind), CPU containers,
# with Prometheus, prometheus-adapter and queue-depth autoscaling.
#
#   ./deploy/up.sh          # then open http://localhost:8080
#   ./deploy/down.sh        # delete the cluster
#
# Needs Docker (8-10 GB of memory for its VM), kind, kubectl and helm.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER=tinyserve
IMAGE=tinyserve:dev
MODEL=$(sed -n 's/^model: \([^ ]*\).*/\1/p' deploy/helm/tinyserve/values.yaml)
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}

for tool in docker kind kubectl helm; do
  command -v "$tool" >/dev/null || { echo "missing: $tool (brew install $tool)"; exit 1; }
done

echo "==> Downloading $MODEL into $HF_CACHE (pods mount this cache instead of downloading)"
mkdir -p "$HF_CACHE"
if command -v uv >/dev/null; then
  HF_HOME="$HF_CACHE" uv run python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', allow_patterns=['*.json', '*.safetensors', '*.jinja'])"
else
  echo "    (uv not found: each pod will download the model on first start)"
fi

echo "==> Cluster"
if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  sed "s|\${HF_CACHE}|$HF_CACHE|" deploy/k8s/kind-cluster.yaml | kind create cluster --config - --wait 2m
fi
kubectl config use-context "kind-$CLUSTER" >/dev/null

echo "==> Image (built for this machine's architecture; see deploy/Dockerfile for multi-arch)"
docker buildx build -f deploy/Dockerfile -t "$IMAGE" --provenance=false --load .
kind load docker-image "$IMAGE" --name "$CLUSTER"

echo "==> Prometheus + prometheus-adapter"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update prometheus-community >/dev/null
helm upgrade --install prometheus prometheus-community/prometheus --version 29.36.1 \
  --namespace monitoring --create-namespace -f deploy/k8s/prometheus-values.yaml --wait
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter --version 5.3.0 \
  --namespace monitoring -f deploy/k8s/prometheus-adapter-values.yaml --wait

echo "==> tinyserve"
helm upgrade --install tinyserve deploy/helm/tinyserve --wait --timeout 10m
# A new image with the same tag isn't picked up by an unchanged Deployment: restart it.
kubectl rollout restart deployment/tinyserve >/dev/null
kubectl rollout status deployment/tinyserve --timeout 10m

cat <<DONE

tinyserve is up.
  UI and API:     http://localhost:8080        (OpenAI base_url: http://localhost:8080/v1)
  Autoscaler:     kubectl get hpa tinyserve --watch
  Load it:        uv run python benchmarks/load_gen.py --url http://localhost:8080 --rate 4 --duration 180
  Tear down:      ./deploy/down.sh
DONE

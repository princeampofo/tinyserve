# tinyserve container image. Build from the repo root:
#   docker buildx build -f deploy/Dockerfile -t tinyserve:dev --load .
#   docker buildx build -f deploy/Dockerfile --platform linux/arm64,linux/amd64 -t <registry>/tinyserve .
# On Linux the lockfile pins CPU-only PyTorch (see pyproject.toml), which keeps
# ~4 GB of CUDA libraries out of the image. For the cloud GPU image, build with
# --build-arg TORCH_BACKEND=cu126 to swap in a CUDA build of the same version.

# ---- 1. the demo UI: compile TypeScript to plain JavaScript
FROM node:22-slim AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/tsconfig.json ./
COPY ui/src ./src
RUN npm run build

# ---- 2. Python dependencies, in a virtualenv the runtime stage copies whole
FROM python:3.12-slim AS python-deps
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /bin/uv
ARG TORCH_BACKEND=""
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# Dependencies first (a cached layer that only rebuilds when uv.lock changes), then the code.
RUN uv sync --frozen --no-dev --no-install-project --no-cache
RUN if [ -n "$TORCH_BACKEND" ]; then \
      VIRTUAL_ENV=/opt/venv uv pip install --no-cache --reinstall --torch-backend "$TORCH_BACKEND" \
        "torch==$(VIRTUAL_ENV=/opt/venv uv pip show torch | sed -n 's/^Version: \([^+]*\).*/\1/p')"; \
    fi
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --no-cache

# ---- 3. runtime: just Python, the virtualenv and the UI files
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
COPY --from=python-deps /opt/venv /opt/venv
COPY ui/index.html ui/style.css /app/ui/
COPY --from=ui /ui/dist /app/ui/dist
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache
USER app
EXPOSE 8000
ENTRYPOINT ["python", "-m", "tinyserve", "--host", "0.0.0.0", "--port", "8000", "--ui-dir", "/app/ui"]
# Small enough for CPU containers; override with --model, --device, --kv-cache-gb, ...
CMD ["--model", "HuggingFaceTB/SmolLM2-360M-Instruct", "--device", "cpu"]

#!/usr/bin/env bash
# Build and push image with inline cache to jlee335/beam-volume-260307
set -euo pipefail

REGISTRY_IMAGE="jlee335/beam-volume-260307:latest"

DOCKER_BUILDKIT=1 docker build \
  --cache-from "${REGISTRY_IMAGE}" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg VLLM_USE_PRECOMPILED=1 \
  --target vllm-openai \
  -t "${REGISTRY_IMAGE}" \
  -f docker/Dockerfile \
  .

docker push "${REGISTRY_IMAGE}"
echo "Pushed cache image to ${REGISTRY_IMAGE}"

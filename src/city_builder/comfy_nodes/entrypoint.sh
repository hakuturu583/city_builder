#!/usr/bin/env bash
# Present whatever weights the mounted cache holds, then be the MCP server.
#
# The linking has to happen at run time, not at build time: the cache is a
# volume, so at build time it is empty. Failure is not fatal — every tool but
# the refinement works without a single model file, and a server that refuses
# to start because a 15 GB download has not happened yet is worse than one that
# says so when asked.
set -u
if [ -x "${COMFYUI_PATH:-/opt/ComfyUI}/link_models.sh" ]; then
    "${COMFYUI_PATH:-/opt/ComfyUI}/link_models.sh" >/dev/null 2>&1 || true
fi
exec city-builder-mcp "$@"

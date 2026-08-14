#!/usr/bin/env bash
# Point ComfyUI at the weights already in the Hugging Face cache.
#
# ComfyUI looks for models under its own tree and Hugging Face keeps them in a
# content-addressed cache, so one of the two has to give. Linking is what gives:
# the cache exposes every blob through `snapshots/` as a real path, so the files
# can be presented in ComfyUI's layout without being downloaded, copied or moved
# — which matters when they are thirty-five gigabytes and shared with everything
# else on the machine that speaks to the hub.
#
#     HF_HOME=/cache /opt/ComfyUI/link_models.sh
#
# Fetch them first, if the cache is empty:
#
#     hf download Abiray/MiniMax-H3-Pruned-GGUF MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf
#     hf download Abiray/MiniMax-H3-GGUF text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
#     hf download Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI \
#         minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors
#     hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors
#     hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_audio_vae_fp32.safetensors
set -euo pipefail

HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
ROOT="${COMFYUI_PATH:-$(cd "$(dirname "$0")" && pwd)}"
missing=0

link() {  # link <models subdir> <file name as it is in the cache>
  local dir="$ROOT/models/$1" name="$2" src
  src="$(find "$HUB" -path '*/snapshots/*' -name "$name" -print -quit 2>/dev/null || true)"
  if [ -z "$src" ]; then
    printf '  %-56s MISSING from %s\n' "$name" "$HUB" >&2
    missing=$((missing + 1))
    return 0
  fi
  mkdir -p "$dir"
  ln -sf "$(readlink -f "$src")" "$dir/$name"
  printf '  %-56s -> models/%s\n' "$name" "$1"
}

link unet           MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf
link loras          minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors
link text_encoders  qwen3vl_32b_minimax_h3-Q4_K_M.gguf
link vae            minimax_h3_video_vae_fp16.safetensors
link vae            minimax_h3_audio_vae_fp32.safetensors

if [ "$missing" -gt 0 ]; then
  echo "$missing file(s) are not in the cache; see the header of this script." >&2
  exit 1
fi

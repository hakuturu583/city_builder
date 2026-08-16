# The MCP server, ready to run over stdio.
#
#     docker run -i --rm --gpus all \
#       -v city-builder-models:/cache -v "$PWD:/work" \
#       ghcr.io/hakuturu583/city_builder-mcp
#
# Two things decide what is in here. `bpy` is a real Blender, so the image needs
# Blender's shared libraries even though nothing ever opens a window — it links
# X11 and GL at import, not at render. And the diffusion stack is in, which is
# most of the 6.9 GB: the texture tools are half of what this server is for, and
# an image that cannot run them is an image whose documentation is hypothetical.
#
# No CUDA base image. The torch wheels carry their own CUDA runtime, so a slim
# Python and `--gpus all` is the whole of it — measured here, `torch.cuda` sees
# the card and paints a facade in 1.7 s. The one exception is building
# TRELLIS.2's CUDA extensions, which is why there is a stage that installs a
# toolkit and nothing that ships it.
#
# One gated model to know about before running the reconstruction: TRELLIS.2
# conditions on DINOv3, whose weights download only for an account that has
# accepted Meta's terms. Pass a token — `-e HF_TOKEN=...`, or a token file on
# the /cache volume. Nothing else here needs an account.
#
# Model *weights* are not in here. They are downloaded, and a few gigabytes
# each, so they belong on the volume at /cache rather than in a layer.
#
# Drop the diffusion stack for a 1.8 GB image without the texture tools:
#
#     docker build --build-arg EXTRAS=mcp -t city-builder-mcp:slim .
#
# Building the CUDA extensions is most of the build time, and most of *that* is
# emitting kernels for five architectures. Building for one is minutes rather
# than the better part of an hour:
#
#     docker build --build-arg TORCH_CUDA_ARCH_LIST=12.0 -t city-builder-mcp .

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Which optional dependency groups go in. Drop `texture comfy` for an image
# without the diffusion stack — a quarter of the size, and four fewer tools.
ARG EXTRAS="mcp texture comfy reconstruct"

WORKDIR /app

# Dependencies first, from the lock file, so a source edit does not re-resolve
# or re-download the several hundred megabytes that bpy weighs.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev \
        $(for extra in $EXTRAS; do printf -- "--extra %s " "$extra"; done)

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
        $(for extra in $EXTRAS; do printf -- "--extra %s " "$extra"; done)


# ComfyUI is a checkout, not a package, so it is fetched rather than installed —
# but its dependencies come from this project's lock file above, which is what
# keeps one torch in the image instead of two.
FROM alpine/git:latest AS comfy

# Pinned: a workflow is a set of node names and input names, and both move.
ARG COMFYUI_REF=7fe8a6138504
ARG COMFYUI_GGUF_REF=main

RUN git clone --filter=blob:none https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout --detach ${COMFYUI_REF} \
    && git clone --depth 1 --branch ${COMFYUI_GGUF_REF} \
        https://github.com/city96/ComfyUI-GGUF.git /opt/ComfyUI/custom_nodes/ComfyUI-GGUF \
    && rm -rf /opt/ComfyUI/.git /opt/ComfyUI/custom_nodes/ComfyUI-GGUF/.git


# TRELLIS.2 is a checkout too, and five CUDA extensions that are not on PyPI.
FROM alpine/git:latest AS trellis

ARG TRELLIS_REF=main
ARG CUMESH_REF=main
ARG FLEXGEMM_REF=main
ARG NVDIFFRAST_REF=v0.4.0
ARG NVDIFFREC_REF=renderutils

RUN git clone --depth 1 --branch ${TRELLIS_REF} \
        https://github.com/microsoft/TRELLIS.2.git /opt/TRELLIS.2 \
    && rm -rf /opt/TRELLIS.2/.git \
    && mkdir -p /src \
    && git clone --depth 1 --recursive --branch ${CUMESH_REF} \
        https://github.com/JeffreyXiang/CuMesh.git /src/CuMesh \
    && git clone --depth 1 --recursive --branch ${FLEXGEMM_REF} \
        https://github.com/JeffreyXiang/FlexGEMM.git /src/FlexGEMM \
    && git clone --depth 1 --branch ${NVDIFFRAST_REF} \
        https://github.com/NVlabs/nvdiffrast.git /src/nvdiffrast \
    && git clone --depth 1 --branch ${NVDIFFREC_REF} \
        https://github.com/JeffreyXiang/nvdiffrec.git /src/nvdiffrec \
    && cp -r /opt/TRELLIS.2/o-voxel /src/o-voxel


# Compiling those five is the one thing here that needs a CUDA toolkit, and it
# is the reason this stage exists rather than the final image growing one: nvcc
# and its headers are three gigabytes that nothing at run time opens. They are
# built into *this project's own* virtualenv, because a torch extension is
# bound to the torch it was compiled against — which is also why they cannot be
# declared as dependencies and resolved.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS extensions

# Must be the same CUDA *major* as the torch the lock file resolved: an
# extension compiled against a different one does not load. The check below
# fails the build rather than letting that surface as an ImportError on a
# machine with a GPU in it.
ARG CUDA_APT=13-0
# Which cards the kernels are emitted for. Ampere through Blackwell by default;
# every architecture is minutes of compilation and megabytes of image, so cut
# this to the card you have if you are building for yourself.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN apt-get update && apt-get install --no-install-recommends -y \
        ca-certificates curl gnupg build-essential git \
        # Eigen is a header-only dependency of o-voxel and is not vendored.
        libeigen3-dev \
    && curl -fsSL -o /tmp/keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/keyring.deb && rm /tmp/keyring.deb \
    && apt-get update \
    # The toolkit only. Never `cuda`, which pulls cuda-drivers and would replace
    # the host driver the container is given.
    && apt-get install --no-install-recommends -y cuda-toolkit-${CUDA_APT} \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app /app
COPY --from=trellis /src /src

ENV PATH="/usr/local/cuda/bin:/app/.venv/bin:$PATH" \
    CPATH=/usr/include/eigen3 \
    UV_LINK_MODE=copy \
    MAX_JOBS=8

RUN --mount=type=cache,target=/root/.cache/uv \
    # A build without the `reconstruct` extra has no torch, and nothing here to
    # compile against. Skipping is the whole difference for the slim image.
    if ! /app/.venv/bin/python -c "import torch" 2>/dev/null; then \
        echo "== no torch in this build; skipping the CUDA extensions"; exit 0; \
    fi \
    && torch_cuda=$(/app/.venv/bin/python -c "import torch; print(torch.version.cuda)") \
    && nvcc_cuda=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p') \
    && echo "== torch built for CUDA ${torch_cuda}, toolkit is ${nvcc_cuda}.x" \
    && case "${torch_cuda}" in "${nvcc_cuda}".*) ;; *) \
        echo "CUDA major mismatch: torch wants ${torch_cuda}, the toolkit is ${nvcc_cuda}.x." \
             "Set --build-arg CUDA_APT to match." >&2; exit 1;; esac \
    && for extension in o-voxel CuMesh FlexGEMM nvdiffrast nvdiffrec; do \
        echo "== building $extension for ${TORCH_CUDA_ARCH_LIST}" \
        && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
           uv pip install --python /app/.venv/bin/python --no-build-isolation \
               "/src/$extension" || exit 1; \
    done


FROM python:3.11-slim-bookworm

# What bpy links against. libgl1/libegl1 and the Mesa drivers are for the
# render tools; without a GPU on the host they fall back to llvmpipe, which is
# slow but correct.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 \
        libsm6 libice6 libgl1 libegl1 libgles2 libglx-mesa0 libgl1-mesa-dri \
        libgomp1 ffmpeg \
        # Triton compiles its kernels when they are first used, so the compiler
        # is a run-time dependency of the GPU path, not a build-time one.
        gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# From `extensions`, not from `build`: it is the same /app with the five
# compiled extensions installed into its virtualenv.
COPY --from=extensions /app /app
COPY --from=comfy /opt/ComfyUI /opt/ComfyUI
COPY --from=trellis /opt/TRELLIS.2 /opt/TRELLIS.2

# The node that lets a sampler start from a render rather than from noise. H3
# ships nothing that does this: its keyframe conditioning is re-injected every
# step and never denoised, so a render handed to it comes back unchanged.
RUN ln -s /app/src/city_builder/comfy_nodes /opt/ComfyUI/custom_nodes/city_builder \
    # The linker writes symlinks here at run time, as whatever user the caller
    # chose — which is not the one that built the image.
    && mkdir -p /opt/ComfyUI/models && chmod -R a+rwX /opt/ComfyUI/models

# ComfyUI looks for weights under its own tree. They live in the mounted cache,
# so the layout points back at it — see the README for the linking script.
COPY --from=build /app/src/city_builder/comfy_nodes/link_models.sh /opt/ComfyUI/link_models.sh
COPY --from=build /app/src/city_builder/comfy_nodes/entrypoint.sh /usr/local/bin/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH" \
    COMFYUI_PATH=/opt/ComfyUI \
    # TRELLIS.2 is a checkout with no package to install, so the pipeline puts
    # this on sys.path rather than importing it by name.
    TRELLIS2_PATH=/opt/TRELLIS.2 \
    PYTHONUNBUFFERED=1 \
    # Model weights are several gigabytes and are downloaded, not shipped, so
    # they must not land in the container's own writable layer — that is thrown
    # away with the container and fetched again on the next run. One env var
    # decides where the Hugging Face cache lives; mount a volume on it.
    HF_HOME=/cache \
    # No GPU in a container by default; let Mesa answer for GL rather than
    # letting Blender fail at import.
    LIBGL_ALWAYS_SOFTWARE=1 \
    # Run as any uid you like — `--user` is how the files you get back stay
    # yours. Torch's compilers write caches under $HOME, which for a uid with no
    # passwd entry is `/`, so they are pointed somewhere every user can write.
    TRITON_CACHE_DIR=/tmp/.triton \
    TORCHINDUCTOR_CACHE_DIR=/tmp/.inductor \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.matplotlib

# Scenes, exports and textures are written wherever the caller asks. Mounting a
# host directory here is what makes those files reachable afterwards.
VOLUME ["/cache"]

WORKDIR /work

# stdio: the client speaks MCP on this process's stdin and stdout, so nothing
# is logged there. `docker run -i` is not optional.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

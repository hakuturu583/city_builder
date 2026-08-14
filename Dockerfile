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
# the card and paints a facade in 1.7 s.
#
# Model *weights* are not in here. They are downloaded, and a few gigabytes
# each, so they belong on the volume at /cache rather than in a layer.
#
# Drop the diffusion stack for a 1.8 GB image without the texture tools:
#
#     docker build --build-arg EXTRAS=mcp -t city-builder-mcp:slim .

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Which optional dependency groups go in. Drop `texture` for an image without
# the diffusion stack — a quarter of the size, and three fewer working tools.
ARG EXTRAS="mcp texture"

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


FROM python:3.11-slim-bookworm

# What bpy links against. libgl1/libegl1 and the Mesa drivers are for the
# render tools; without a GPU on the host they fall back to llvmpipe, which is
# slow but correct.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 \
        libsm6 libice6 libgl1 libegl1 libgles2 libglx-mesa0 libgl1-mesa-dri \
        libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Model weights are several gigabytes and are downloaded, not shipped, so
    # they must not land in the container's own writable layer — that is thrown
    # away with the container and fetched again on the next run. One env var
    # decides where the Hugging Face cache lives; mount a volume on it.
    HF_HOME=/cache \
    # No GPU in a container by default; let Mesa answer for GL rather than
    # letting Blender fail at import.
    LIBGL_ALWAYS_SOFTWARE=1

# Scenes, exports and textures are written wherever the caller asks. Mounting a
# host directory here is what makes those files reachable afterwards.
VOLUME ["/cache"]

WORKDIR /work

# stdio: the client speaks MCP on this process's stdin and stdout, so nothing
# is logged there. `docker run -i` is not optional.
ENTRYPOINT ["city-builder-mcp"]

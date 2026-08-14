# The MCP server, ready to run over stdio.
#
#     docker run -i --rm -v "$PWD:/work" ghcr.io/hakuturu583/city_builder-mcp
#
# Two things decide what is in here. `bpy` is a real Blender, so the image needs
# Blender's shared libraries even though nothing ever opens a window — it links
# X11 and GL at import, not at render. And the diffusion extra is left out: it
# would add several gigabytes of CUDA that cannot do anything without a GPU on
# the host anyway, so the geometry, export, survey and render tools are what
# this image serves, and `generate_facades` reports that its weights are
# missing rather than pretending.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, from the lock file, so a source edit does not re-resolve
# or re-download the several hundred megabytes that bpy weighs.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra mcp

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra mcp


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
    # No GPU in a container by default; let Mesa answer for GL rather than
    # letting Blender fail at import.
    LIBGL_ALWAYS_SOFTWARE=1

# Scenes, exports and textures are written wherever the caller asks. Mounting a
# host directory here is what makes those files reachable afterwards.
WORKDIR /work

# stdio: the client speaks MCP on this process's stdin and stdout, so nothing
# is logged there. `docker run -i` is not optional.
ENTRYPOINT ["city-builder-mcp"]

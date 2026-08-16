"""Build ground and road-surface meshes from a Lanelet2 HD map.

    from city_builder import build_city, build_scene

    result = build_city("lanelet2_map.osm")
    build_scene(result, blend="scene.blend", glb="scene.glb")

A Lanelet2 map ships the surveyed left and right boundary of every lane in 3D,
so the road surface is read rather than inferred — true width, true curvature,
true elevation, real turning lanelets at the intersections. It ships no terrain
at all, so the ground is reconstructed from the road elevations and clipped to
the kerb line (see :mod:`city_builder.ground`).
"""

import os as _os

# Before anything imports torch, because this is read once — when the CUDA
# caching allocator is first initialised — and ignored ever after. Two places
# in this package set it on the way into a model, and both are too late in a
# pipeline run: SDXL draws the ground tiles and the building photographs first,
# and by the time TRELLIS.2 asks for expandable segments the allocator has
# already been built without them. What that costs is buildings: the mesher
# wants one large contiguous block and a fragmented arena cannot give it one,
# and five plots in sixty died of it with 25 GB of the card free.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .build import BuildResult, build_city, build_scene, write_heightmap, write_manifest
from .buildings import BuildingOptions
from .classes import CLASSES, GENERATE, PRESERVE, SurfaceClass
from .facade_layout import FacadeLayout
from .frame import LocalFrame
from .geometry import Mesh, Polygon, Ribbon
from .ground import HeightMap
from .surfaces import SurfaceOptions

__all__ = [
    "CLASSES",
    "GENERATE",
    "PRESERVE",
    "BuildResult",
    "BuildingOptions",
    "FacadeLayout",
    "HeightMap",
    "LocalFrame",
    "Mesh",
    "Polygon",
    "Ribbon",
    "SurfaceClass",
    "SurfaceOptions",
    "build_city",
    "build_scene",
    "write_heightmap",
    "write_manifest",
]

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

from .build import BuildResult, build_city, build_scene, write_heightmap
from .frame import LocalFrame
from .geometry import Mesh, Polygon, Ribbon
from .ground import HeightMap
from .surfaces import SurfaceOptions

__all__ = [
    "BuildResult",
    "HeightMap",
    "LocalFrame",
    "Mesh",
    "Polygon",
    "Ribbon",
    "SurfaceOptions",
    "build_city",
    "build_scene",
    "write_heightmap",
]

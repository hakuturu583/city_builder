"""What a video model has to be told about a drive, taken from the scene itself.

A depth-conditioned video model needs a depth video and, if what comes back is
going to be distilled into a 3D representation, the camera that saw each frame.
Both are usually *estimated* — a monocular depth network, then structure from
motion — because the usual input is a recording of somewhere real. Here the
scene is ours. The depth is not inferred, it is measured off the geometry that
was rendered, and the camera is not solved for, it is the one that was keyed.

**Depth as colour, not as a render pass.** The same reason ``masks.py`` renders
object identity that way: the compositor moved in Blender 5 and the depth pass
has never been dependable outside Cycles. An emission shader whose value *is*
the distance to the camera works in EEVEE, at any sample count, and lands in a
float EXR in metres. Measured against a scene of known height: a camera 60 m
above the ground read 59.9-60.2 m on the ground and 44.2 m on the tallest roof,
which is that building's 15.8 m.

**Planar depth, not radial.** ``View Z Depth`` is the distance along the view
axis, so a flat floor reads the same at the edge of frame as at the centre.
That is what a depth-conditioned model is trained on and what unprojecting to a
point cloud wants; radial distance would put a bowl in every flat surface.

**One sample, no filter, no dither.** All three for the same reason as the ID
pass, and it matters more here: EEVEE jitters the camera between samples and
averages, so a silhouette pixel comes back as the mean of the near surface and
the far one — a depth that is on neither, floating in the air between them.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Blender's camera looks down its local -Z with +Y up. Nearly everything that
#: consumes a pose — OpenCV, gsplat, the 3DGS reference implementation — looks
#: down +Z with +Y down. This is the change of basis between them, and it is a
#: rotation: `diag(1, -1, -1)` has determinant +1, so no handedness is lost.
BLENDER_TO_CV = np.diag([1.0, -1.0, -1.0])

#: Blender's default camera sensor. `lens` is in millimetres against this.
SENSOR_MM = 36.0


@dataclass
class Camera:
    """One frame's camera, in the convention a rasteriser expects."""

    frame: int
    #: 4x4 world-to-camera, +Z forward, +Y down.
    view: np.ndarray
    #: 3x3 pinhole intrinsics in pixels.
    intrinsics: np.ndarray
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame,
                "view": [[round(float(v), 8) for v in row] for row in self.view],
                "intrinsics": [[round(float(v), 6) for v in row] for row in self.intrinsics],
                "width": self.width, "height": self.height}

    @property
    def position(self) -> np.ndarray:
        """Where the camera is, back in world coordinates."""
        rotation, translation = self.view[:3, :3], self.view[:3, 3]
        return -rotation.T @ translation


def intrinsics(width: int, height: int, lens_mm: float,
               sensor_mm: float = SENSOR_MM) -> np.ndarray:
    """Pinhole intrinsics for a Blender camera of this focal length.

    Blender's default sensor fit is ``AUTO``, which puts the sensor across the
    *larger* of the two image dimensions. Assuming it is always the width gives
    a focal length that is wrong by the aspect ratio on any portrait render, and
    the error shows up as a depth that unprojects into a scene the wrong size.
    """
    across = max(width, height)
    focal = across * lens_mm / sensor_mm
    return np.array([[focal, 0.0, width / 2.0],
                     [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]])


def world_to_camera(matrix_world) -> np.ndarray:
    """A Blender camera's ``matrix_world`` as a world-to-camera matrix.

    Taken from the object rather than rebuilt from the ``(position, target)``
    the route handed over: ``to_track_quat`` has behaviour at the poles that is
    Blender's to define, and a pose that disagrees with the frame it labels is
    worse than no pose at all.
    """
    matrix = np.asarray(matrix_world, dtype=float).reshape(4, 4)
    to_world = matrix[:3, :3] @ BLENDER_TO_CV
    position = matrix[:3, 3]

    view = np.eye(4)
    view[:3, :3] = to_world.T
    view[:3, 3] = -to_world.T @ position
    return view


def unproject(depth: np.ndarray, camera: Camera) -> np.ndarray:
    """Planar depth back into world points, one per pixel. The check on the rest.

    If the depth, the intrinsics and the pose agree, these land on the surfaces
    that were rendered. If any one of them is wrong they land somewhere plausible
    and nothing else notices, which is why this is here rather than in a caller.
    """
    height, width = depth.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width]
    fx, fy = camera.intrinsics[0, 0], camera.intrinsics[1, 1]
    cx, cy = camera.intrinsics[0, 2], camera.intrinsics[1, 2]

    local = np.stack([(xs - cx) / fx * depth,
                      (ys - cy) / fy * depth,
                      depth], axis=-1)
    rotation, translation = camera.view[:3, :3], camera.view[:3, 3]
    return (local - translation) @ rotation


def write_cameras(path: str, cameras: list[Camera], *, extra: dict | None = None) -> str:
    """The poses as one JSON, because a drive is one thing."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {"convention": "world-to-camera, +Z forward, +Y down, pixels",
               "frames": [camera.to_dict() for camera in cameras]}
    payload.update(extra or {})
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
    return path


def read_cameras(path: str) -> list[Camera]:
    """Read them back, for anything downstream that only has the files."""
    with open(path) as handle:
        payload = json.load(handle)
    return [Camera(frame=int(f["frame"]),
                   view=np.asarray(f["view"], dtype=float),
                   intrinsics=np.asarray(f["intrinsics"], dtype=float),
                   width=int(f["width"]), height=int(f["height"]))
            for f in payload["frames"]]


# ---------------------------------------------------------------------------
# The render. Everything below wants bpy.
# ---------------------------------------------------------------------------


def depth_material(name: str = "ViewDepth"):
    """An emission whose value is the distance along the view axis, in metres."""
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    camera_data = tree.nodes.new("ShaderNodeCameraData")
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(camera_data.outputs["View Z Depth"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def paint_depth(objects=None) -> int:
    """Put that material on every mesh. Destructive, like the ID pass.

    This is a second run over a scene that has already been rendered for
    beauty; the caller reopens the .blend rather than putting the materials
    back.
    """
    import bpy

    material = depth_material()
    painted = 0
    for obj in list(objects if objects is not None else bpy.data.objects):
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)
        painted += 1
    return painted


def to_linear(value: float) -> float:
    """The scene-linear value that comes out of the display transform as ``value``.

    A shader colour is scene-linear and an 8-bit render is written through the
    sRGB transform, so a colour put straight into an emission comes back as a
    different one. ``masks.py`` measured it: 7/255 in became 34 out. Only the
    byte-per-channel passes need this — the depth is a float EXR and is not
    touched by it.
    """
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def class_colours() -> dict[str, tuple[float, float, float, float]]:
    """The emission colour per surface class, pre-compensated for the transform.

    The colours are the ones ``classes.py`` already carries. They were chosen to
    be told apart by eye in a debug render, which is the same property a
    segmentation control image needs, so there is no second palette here.
    """
    from .classes import CLASSES

    return {name: (*(to_linear(c) for c in surface.mask_colour), 1.0)
            for name, surface in CLASSES.items()}


def paint_classes(objects=None) -> dict[str, str]:
    """Flat emission per object, keyed to the class it belongs to.

    Returns what was painted what, because a control image nothing can name is
    not much use. An object with no ``cb_class`` — a reconstruction dropped into
    the scene, say — takes the class of the group it stands in if it can be
    read off the name, and black otherwise.
    """
    import bpy

    palette = class_colours()
    painted: dict[str, str] = {}
    for obj in list(objects if objects is not None else bpy.data.objects):
        if obj.type != "MESH":
            continue
        name = str(obj.get("cb_class") or "")
        if name not in palette and name != "":
            name = ""
        if not name:
            # `rebuilt_b0002_geometry_0` is a building, and says so in its name.
            name = "Buildings" if obj.name.startswith("rebuilt_") else ""

        colour = palette.get(name, (0.0, 0.0, 0.0, 1.0))
        material = bpy.data.materials.new(f"Class_{obj.name}")
        material.use_nodes = True
        tree = material.node_tree
        tree.nodes.clear()
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = colour
        emission.inputs["Strength"].default_value = 1.0
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        obj.data.materials.clear()
        obj.data.materials.append(material)
        painted[obj.name] = name or "(unclassified)"
    return painted


def flatten_for_segmentation() -> None:
    """As for depth, but written as bytes, so the colours must survive that."""
    import bpy

    flatten_for_depth()
    render = bpy.context.scene.render
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGB"
    render.image_settings.color_depth = "8"
    render.image_settings.compression = 0


def render_segmentation(out_dir: str, *, frames: int, resolution: tuple[int, int],
                        verbose: bool = True) -> list[str]:
    """The keyed camera path again, as which class every pixel belongs to.

    This is the conditioning the depth cannot give. A road surface is one flat
    plane in depth, so the paint on it carries no signal at all and a
    depth-conditioned model invents its own lane markings — measured, and the
    reason this exists. Built with ``--marking-geometry``, ``LaneMarkings`` is
    an object of its own and lands here as its own colour.
    """
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    painted = paint_classes()
    flatten_for_segmentation()
    if verbose:
        seen = sorted(set(painted.values()))
        print(f"[segmentation] {len(painted)} object(s) over {len(seen)} class(es): "
              f"{', '.join(seen)}")

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
        path = os.path.join(out_dir, f"class_{frame:05d}.png")
        scene.render.filepath = path[: -len(".png")]
        bpy.ops.render.render(write_still=True)
        written.append(path)
        if verbose and frame % 30 == 0:
            print(f"[segmentation] {frame}/{frames}")
    return written


def flatten_for_depth() -> None:
    """Take away everything that could tint or blend a distance.

    The world goes entirely rather than to black — ``masks.py`` found that
    setting it black left the sky shader running — and with it the lights, the
    display transform, the reconstruction filter, the dither and every sample
    but one. The last is the one that matters most for depth: EEVEE jitters the
    camera between samples and averages the result, so a silhouette pixel comes
    back as the mean of the surface in front and the surface behind, which is a
    distance to neither.
    """
    import bpy

    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.world = None

    view = scene.view_settings
    view.view_transform = "Standard"
    view.look = "None"
    view.exposure = 0.0
    view.gamma = 1.0

    render = scene.render
    render.filter_size = 0.0
    render.dither_intensity = 0.0
    scene.eevee.taa_render_samples = 1
    render.film_transparent = False
    # Float EXR: a depth squashed into a byte is a depth quantised to the
    # nearest 15 cm over a 40 m street, and every consumer wants a different
    # normalisation anyway. Metres now, whatever the model wants later.
    render.image_settings.file_format = "OPEN_EXR"
    render.image_settings.color_mode = "RGB"
    render.image_settings.color_depth = "32"
    render.image_settings.exr_codec = "ZIP"


def cameras_along(path_length: int, width: int, height: int) -> list[Camera]:
    """The scene camera's pose at each frame of the animation it is keyed to."""
    import bpy

    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        raise RuntimeError("no camera in the scene to read poses from")

    lens = camera.data.lens
    matrix = intrinsics(width, height, lens)
    out = []
    for frame in range(1, path_length + 1):
        scene.frame_set(frame)
        out.append(Camera(frame=frame,
                          view=world_to_camera(camera.matrix_world),
                          intrinsics=matrix, width=width, height=height))
    return out


def render_depth(out_dir: str, *, frames: int, resolution: tuple[int, int],
                 verbose: bool = True) -> list[str]:
    """Render the keyed camera path again, as distance. One EXR per frame."""
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    painted = paint_depth()
    flatten_for_depth()
    if verbose:
        print(f"[depth] {painted} mesh object(s) painted with view distance")

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
        path = os.path.join(out_dir, f"depth_{frame:05d}.exr")
        scene.render.filepath = path[: -len(".exr")]
        bpy.ops.render.render(write_still=True)
        written.append(path)
        if verbose and frame % 30 == 0:
            print(f"[depth] {frame}/{frames}")
    return written


def sky(depth: np.ndarray) -> np.ndarray:
    """Where the ray hit nothing. Zero metres means *no surface*, not "here".

    An emission shader returns 0 where there is no geometry, so the sky comes
    back as a depth of zero — which is the nearest value there is. Handed
    straight to a model that reads near as bright, the sky becomes the closest
    thing in the frame and the restyled result puts a wall across the horizon.
    Roughly a tenth of a street-level frame is sky, so this is not a corner.
    """
    return ~np.isfinite(depth) | (depth <= 0.0)


def to_control_image(depth: np.ndarray, *, near: float | None = None,
                     far: float | None = None, invert: bool = True) -> np.ndarray:
    """Metric depth as the 0-1 image a depth-conditioned model is trained on.

    Inverted by default — near is bright — because that is what the depth
    estimators these models were conditioned on produce, and a model shown the
    other sense reads a street as a tunnel.

    ``near`` and ``far`` default to the range actually present, excluding the
    sky. Fixing them across a sequence is usually what you want instead: a
    per-frame range makes the brightness of a given wall change as other things
    enter and leave the shot, and that flicker is exactly what a video model
    will faithfully reproduce.
    """
    nothing = sky(depth)
    metres = np.where(nothing, np.nan, depth)
    if not np.isfinite(metres).any():
        return np.zeros_like(depth, dtype=np.float32)

    low = float(np.nanmin(metres)) if near is None else float(near)
    high = float(np.nanmax(metres)) if far is None else float(far)
    high = max(high, low + 1e-6)

    if invert:
        # In inverse depth, so the resolution goes where the detail is.
        one = 1.0 / np.clip(metres, low, high)
        image = (one - 1.0 / high) / (1.0 / low - 1.0 / high)
    else:
        image = (np.clip(metres, low, high) - low) / (high - low)

    # The sky is the far end, whichever way round the ramp runs.
    return np.where(nothing, 0.0, image).astype(np.float32)


def depth_range(frames, *, low: float = 0.5, high: float = 99.5,
                budget: int = 2_000_000) -> tuple[float, float]:
    """One near and far for a whole drive, from the depths that are in it.

    Guessing the range wastes most of it. Inverse depth puts its resolution at
    the near end, so a ``near`` of 1 m on a street whose closest surface is the
    road four metres ahead spends three quarters of the ramp on distances that
    never occur — measured on a 2 s drive, every real surface landed inside the
    darkest quarter of the image.

    Percentiles rather than the extremes, for the same reason ``fitting.bounds``
    uses them: one stray polygon at the clip plane should not set the far end.
    """
    kept = []
    for depth in frames:
        values = depth[~sky(depth)]
        if values.size:
            kept.append(values)
    if not kept:
        raise ValueError("every frame is sky; there is no depth to range")

    values = np.concatenate(kept)
    if values.size > budget:                       # hundreds of frames of 720p
        values = values[:: values.size // budget + 1]
    return float(np.percentile(values, low)), float(np.percentile(values, high))


def read_depth(path: str) -> np.ndarray:
    """One EXR back as metres. Single channel — the three are identical."""
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not read {path} as an EXR")
    return (image[..., 0] if image.ndim == 3 else image).astype(np.float32)


def reproject(camera: Camera, depth: np.ndarray, sources, *,
              tolerance: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """``(colour, valid)`` for one view, gathered from views already rendered.

    What it is for: carrying a picture from cameras that have one into a camera
    that does not. A video model asked to continue a drive needs to be shown
    what the last stretch looked like, and the only honest way to put that in
    front of the new camera is to move it there through the geometry — which is
    measured here rather than estimated, so this is arithmetic rather than a
    guess.

    ``sources`` is a sequence of ``(camera, depth, rgb)`` already in hand, most
    recent first; each target pixel takes the first answer that survives.

    **Backward, not forward.** Scattering source pixels into the target leaves
    holes wherever the sampling stretches, and a hole is indistinguishable from
    surface that genuinely was not seen. Going the other way every target pixel
    asks a question and either gets an answer or does not, and "does not" is
    exactly the mask a caller wants.

    **Occlusion is checked.** A world point can land inside an old frame and
    still have been hidden there, behind something since driven past. If the old
    depth at that pixel disagrees with how far the point actually is — by more
    than ``tolerance`` of the distance, because a fixed margin is wrong at both
    two metres and eighty — the answer is thrown away.
    """
    height, width = depth.shape[:2]
    world = unproject(depth, camera)
    colour = np.zeros((height, width, 3), np.float32)
    valid = np.zeros((height, width), bool)
    nothing = sky(depth)

    for source, source_depth, rgb in sources:
        want = ~valid & ~nothing
        if not want.any():
            break
        local = world[want] @ source.view[:3, :3].T + source.view[:3, 3]
        ahead = local[:, 2] > 0.05
        fx, fy = source.intrinsics[0, 0], source.intrinsics[1, 1]
        cx, cy = source.intrinsics[0, 2], source.intrinsics[1, 2]

        u = np.full(len(local), -1.0)
        v = np.full(len(local), -1.0)
        u[ahead] = local[ahead, 0] / local[ahead, 2] * fx + cx
        v[ahead] = local[ahead, 1] / local[ahead, 2] * fy + cy
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        inside = (ahead & (ui >= 0) & (ui < source.width)
                  & (vi >= 0) & (vi < source.height))
        if not inside.any():
            continue

        there = np.zeros(len(local))
        there[inside] = source_depth[vi[inside], ui[inside]]
        agrees = np.zeros(len(local), bool)
        near = inside & (there > 0)
        agrees[near] = np.abs(there[near] - local[near, 2]) < tolerance * local[near, 2]
        if not agrees.any():
            continue

        rows, cols = np.nonzero(want)
        colour[rows[agrees], cols[agrees]] = rgb[vi[agrees], ui[agrees]]
        valid[rows[agrees], cols[agrees]] = True

    return colour, valid


def rings_of(shapes) -> list[np.ndarray]:
    """Every shape in a build group as closed world-space rings.

    A ribbon becomes one ring by walking its left bound and coming back down
    its right; a polygon is already one; an infill is triangulated, and each
    face goes separately so the hole in a junction stays a hole.
    """
    out: list[np.ndarray] = []
    for shape in shapes:
        if hasattr(shape, "left"):
            found = [list(shape.left) + list(shape.right)[::-1]]
        elif hasattr(shape, "points"):
            found = [list(shape.points)]
        else:
            found = [[shape.vertices[i] for i in face] for face in shape.faces]
        out.extend(np.asarray(ring, dtype=float) for ring in found if len(ring) >= 3)
    return out


def cover(rings, camera: Camera, depth: np.ndarray | None = None, *,
          tolerance: float = 0.25, scale: int = 2) -> np.ndarray:
    """Which pixels a set of world-space rings covers, seen from a camera.

    For surfaces the build paints into a texture rather than leaving as objects
    of their own — lane markings above all — which is why they cannot simply be
    read back out of a class render.

    Filled at ``scale`` times the frame and reduced, because a lane line forty
    metres away is thinner than a pixel and an aliased mask makes it dash.

    ``depth`` is the frame's depth pass. Given it, a ring hidden behind
    something is dropped. The distance to compare against is worked out per
    pixel, from where that pixel's ray crosses the ring's own plane — one
    distance for the whole ring is only right for a small one, and a lane
    marking is twenty metres long, so it takes the near end of every line with
    it.
    """
    from PIL import Image, ImageDraw

    size = (camera.width * scale, camera.height * scale)
    matrix = camera.intrinsics * np.array([[scale], [scale], [1.0]])
    rotation, translation = camera.view[:3, :3], camera.view[:3, 3]
    centre = -rotation.T @ translation

    u, v = np.meshgrid(np.arange(size[0]) + 0.5, np.arange(size[1]) + 0.5)
    rays = np.linalg.inv(matrix) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    forward = (rotation.T @ rays).T

    wanted = None
    if depth is not None:
        wanted = np.asarray(
            Image.fromarray(depth).resize(size, Image.NEAREST), np.float32).ravel()

    mask = np.zeros(u.size, bool)
    for ring in rings:
        local = (rotation @ ring.T).T + translation
        if (local[:, 2] <= 0.05).any():
            continue
        pixels = (matrix @ local.T).T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        if (pixels[:, 0].max() < 0 or pixels[:, 1].max() < 0
                or pixels[:, 0].min() > size[0] or pixels[:, 1].min() > size[1]):
            continue

        patch = Image.new("1", size, 0)
        ImageDraw.Draw(patch).polygon([tuple(p) for p in pixels], fill=1)
        here = np.array(patch, bool).ravel()
        if not here.any():
            continue

        if wanted is not None:
            plane = float(ring[:, 2].mean())
            with np.errstate(divide="ignore", invalid="ignore"):
                along = (plane - centre[2]) / forward[:, 2]
            reach = along * rays[2]
            here &= np.isfinite(reach) & (along > 0) & (reach <= wanted + tolerance)
        mask |= here

    reduced = mask.reshape(camera.height, scale, camera.width, scale)
    return reduced.mean(axis=(1, 3))

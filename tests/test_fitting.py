"""Placing a generated mesh on a plot it never saw.

The failure this guards against is silent: a building that is scaled and turned
plausibly, and sits three metres into the pavement. Nothing errors, the render
looks fine from the far side, and the road geometry the whole scene exists to
specify is wrong.
"""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point, Polygon

from city_builder.fitting import apply, fit_to_plot, overhang


def box(width: float, depth: float, height: float, *, up: str = "y"):
    """Unit-ish mesh corners the way an image-to-3D model returns them."""
    half_w, half_d = width / 2, depth / 2
    corners = [(x, y, z) for x in (-half_w, half_w)
               for y in (-half_d, half_d) for z in (0.0, height)]
    points = np.asarray(corners, dtype=float)
    if up == "y":  # swap so the tall axis is y, as Hunyuan3D answers
        points = points[:, [0, 2, 1]]
    return points


def plot(long_side: float, short_side: float, heading_deg: float,
         centre=(10.0, 20.0)) -> Polygon:
    angle = math.radians(heading_deg)
    along = np.array([math.cos(angle), math.sin(angle)])
    across = np.array([-along[1], along[0]])
    middle = np.asarray(centre, dtype=float)
    corners = [middle + along * s * long_side / 2 + across * t * short_side / 2
               for s, t in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return Polygon([tuple(c) for c in corners])


def test_a_generated_box_lands_on_its_plot():
    mesh = box(0.8, 0.4, 1.0)
    footprint = plot(24.0, 12.0, 35.0)
    placement = fit_to_plot(mesh, footprint, base_z=3.0, height=21.0)
    placed = apply(mesh, placement)

    assert placed[:, 2].min() == 3.0                    # stood on the pavement
    assert abs(placed[:, 2].max() - 24.0) < 1e-6        # 3 m base + 21 m tall
    covered = Polygon(footprint).buffer(0.01)
    assert all(covered.covers(Point(p[:2])) for p in placed)


def handedness(points, a: int, b: int, c: int, d: int) -> float:
    """Six times the signed volume of a tetrahedron: its *sign* is the chirality."""
    p = np.asarray(points, dtype=float)
    return float(np.linalg.det(np.stack([p[b] - p[a], p[c] - p[a], p[d] - p[a]])))


def test_the_mesh_is_rotated_into_the_scene_and_never_mirrored():
    """y-up to z-up is a rotation. Getting it wrong is invisible everywhere else.

    ``(x, y, z) -> (x, z, y)`` looks like the swap the job needs and is a
    reflection: the building stands up, fills its plot and scores exactly as
    well, and its signage reads backwards. Nothing downstream of here has an
    opinion about chirality, so it has to be tested at the transform.

    A symmetric box cannot show it — which is what every other test in this file
    uses — so this one measures the sign of a tetrahedron's volume instead.
    """
    mesh = box(0.8, 0.4, 1.0)
    footprint = plot(24.0, 12.0, 35.0)
    placed = apply(mesh, fit_to_plot(mesh, footprint, base_z=0.0, height=21.0))

    before = handedness(mesh, 0, 1, 2, 4)
    after = handedness(placed, 0, 1, 2, 4)
    assert abs(before) > 1e-9                      # the corners are not coplanar
    assert math.copysign(1.0, after) == math.copysign(1.0, before)


def test_the_wing_of_an_asymmetric_building_stays_on_its_own_side():
    """The same property said as a building: a mirrored plan is still a plan."""
    mesh = box(0.8, 0.4, 1.0)
    # A wing off one corner, so the plan has a left and a right at all.
    wing = np.array([[0.4, 0.0, 0.2], [0.4, 1.0, 0.2]])   # y is up in this frame
    mesh = np.vstack([mesh, wing])
    footprint = plot(24.0, 12.0, 0.0)
    placed = apply(mesh, fit_to_plot(mesh, footprint, base_z=0.0, height=21.0))

    # The wing is at +x and +z in the mesh; z-up rotation sends +z to -y.
    assert placed[-1][0] > 0.0 + 10.0              # plot centre is (10, 20)
    assert placed[-1][1] < 20.0


def test_the_long_side_of_the_mesh_meets_the_long_side_of_the_plot():
    # A mesh whose long axis is depth rather than width has to be turned a
    # quarter turn, or a 24 m building is fitted across a 12 m frontage.
    upright = fit_to_plot(box(0.8, 0.4, 1.0), plot(24.0, 12.0, 0.0), 0.0, 21.0)
    sideways = fit_to_plot(box(0.4, 0.8, 1.0), plot(24.0, 12.0, 0.0), 0.0, 21.0)
    assert abs((sideways.yaw - upright.yaw) - math.pi / 2) < 1e-9
    assert np.allclose(upright.scale, sideways.scale)


def test_a_wrongly_proportioned_mesh_is_reported_not_hidden():
    # Squat mesh, tall plot: the fit still works, and says how hard it pulled.
    placement = fit_to_plot(box(1.0, 1.0, 0.2), plot(20.0, 20.0, 0.0), 0.0, 40.0)
    assert placement.distortion > 5.0


def test_uniform_scaling_keeps_proportions_and_then_overhangs():
    footprint = plot(24.0, 12.0, 0.0)
    mesh = box(0.8, 0.8, 1.0)   # square in plan, plot is not
    stretched = apply(mesh, fit_to_plot(mesh, footprint, 0.0, 21.0))
    kept = apply(mesh, fit_to_plot(mesh, footprint, 0.0, 21.0, uniform=True))
    assert overhang(stretched, footprint)["worst_overhang_m"] < 1e-6
    # Keeping the proportions is exactly what puts it over the boundary, which
    # is why it is not the default.
    assert overhang(kept, footprint)["worst_overhang_m"] > 1.0


def test_heading_is_read_from_the_footprint_not_the_bounding_box():
    # A plot at forty degrees to north: its bounding box is much bigger than it
    # is, and fitting to that would leave the building crossing the boundary.
    footprint = plot(30.0, 10.0, 40.0)
    mesh = box(0.9, 0.3, 1.0)
    placed = apply(mesh, fit_to_plot(mesh, footprint, 0.0, 18.0))
    assert overhang(placed, footprint)["worst_overhang_m"] < 1e-6

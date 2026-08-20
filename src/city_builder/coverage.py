"""How much of a cloud a set of drives has actually looked at, and what to drive next.

A splat nobody's camera reached never got a gradient, so it still carries the
flat colour the mesh gave it. Coverage is therefore not a statistic about the
process — it is the fraction of the asset that exists. One forward camera down
one street of the t-junction reached 55%.

The cure is not more route: on a map this size every seed finds the same
thirty-eight metres. It is more *direction*. A wall beside the road and the
pavement under it are only ever seen edge-on from a car looking where it is
going, and turning the camera is what puts them in front of it — measured, the
first two sideways passes were worth thirty points between them, and the four
after that eight between them.

So the passes here work outwards from straight ahead, alternating sides. The
order matters because a sweep is stopped when it stops paying, and stopping
early should leave a balanced set rather than everything the drive saw to its
left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pass:
    """One drive of the same route, looking somewhere other than ahead."""

    #: Degrees off the direction of travel, anticlockwise.
    yaw: float = 0.0
    #: Metres across the lane, positive to the left.
    sideways: float = 0.0

    @property
    def name(self) -> str:
        return f"yaw{round(self.yaw):+04d}_x{self.sideways:+.1f}"


#: Outwards from straight ahead, alternating sides. The last four move the
#: camera across the lane instead of turning it, which is what is left once
#: turning has run out of new surfaces to find.
SWEEP = (
    Pass(0.0), Pass(-55.0), Pass(55.0), Pass(-110.0), Pass(110.0), Pass(180.0),
    Pass(0.0, -2.0), Pass(0.0, 2.0), Pass(-75.0, 1.5), Pass(75.0, -1.5),
)


def turned(path, look: Pass):
    """The same drive, pointed somewhere else.

    The target is swung about the camera rather than the camera about the
    target: a driver who glances left is still in the same lane, and moving the
    camera instead would drive a different route.
    """
    yaw = math.radians(look.yaw)
    out = []
    for eye, target in path:
        eye = np.asarray(eye, dtype=float)
        reach = np.asarray(target, dtype=float) - eye
        forward = reach[:2] / max(float(np.linalg.norm(reach[:2])), 1e-9)
        side = np.array([-forward[1], forward[0]])
        spun = np.array([reach[0] * math.cos(yaw) - reach[1] * math.sin(yaw),
                         reach[0] * math.sin(yaw) + reach[1] * math.cos(yaw),
                         reach[2]])
        moved = eye + np.array([side[0] * look.sideways,
                                side[1] * look.sideways, 0.0])
        out.append((tuple(moved), tuple(moved + spun)))
    return out


def enough(history, *, target: float, least: float) -> str | None:
    """Why a sweep should stop, or None to keep going.

    Two reasons, and they are different. Reaching the target is success. A pass
    that adds less than `least` is the sweep telling you that turning the camera
    has stopped finding new surfaces, and the answer to that is a different
    route rather than another look down this one.
    """
    if not history:
        return None
    if history[-1]["coverage"] >= target:
        return f"target reached: {history[-1]['coverage']:.1f}% >= {target:.1f}%"
    if len(history) > 1 and history[-1]["gained"] < least:
        return (f"a pass added {history[-1]['gained']:.1f}%, below {least:.1f}%; "
                f"stopping at {history[-1]['coverage']:.1f}%")
    return None

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
    """One drive: which road, how high, and where it is looking."""

    #: Which of the covering routes to drive.
    route: int = 0
    #: Degrees off the direction of travel, anticlockwise.
    yaw: float = 0.0
    #: Metres across the lane, positive to the left.
    sideways: float = 0.0
    #: Metres above the road. A driver's eye by default; higher sees over
    #: parked cars and hedges, and down onto surfaces a car never looks at.
    height: float = 1.5
    #: Drive the route the other way. A splat carries one colour, so a surface
    #: only ever approached from one end is fixed at how it looked from that
    #: end — and the far side of a pole, a tree or a kerb is never seen at all.
    reverse: bool = False

    @property
    def name(self) -> str:
        way = "rev" if self.reverse else "fwd"
        return (f"r{self.route}{way}_yaw{round(self.yaw):+04d}"
                f"_x{self.sideways:+.1f}_z{self.height:.1f}")


#: What one route is worth looking at, outwards from straight ahead and
#: alternating sides. The raised pass is last of the turns because height buys
#: less than direction does — a wall is still a wall from half a metre up — but
#: it is the only thing that sees the top of anything.
LOOKS = (
    Pass(yaw=0.0), Pass(yaw=-55.0), Pass(yaw=55.0),
    Pass(yaw=-110.0), Pass(yaw=110.0), Pass(yaw=180.0),
    Pass(yaw=-40.0, height=4.0), Pass(yaw=40.0, height=4.0),
)


def sweep(routes: int, looks=LOOKS, *, both_ways: bool = True) -> list[Pass]:
    """Every route seen every way, but the roads first, and both ways early.

    Ordered by look, then by direction, then by route — not the other way
    round. A sweep is stopped when it stops paying, and driving one street eight
    ways before touching the next would leave whole roads at the flat colour the
    mesh gave them while the first was being polished.

    Both directions come before any turning because they are worth more. A
    splat carries a single colour, so a wall approached from one end only is
    fixed at how it looked from that end, and the far face of anything standing
    on the pavement is never seen at all.
    """
    ways = (False, True) if both_ways else (False,)
    return [Pass(route=index, yaw=look.yaw, sideways=look.sideways,
                 height=look.height, reverse=backwards)
            for look in looks for backwards in ways for index in range(routes)]


#: The default sweep, for a map with a single route worth driving.
SWEEP = tuple(sweep(1))


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


def enough(history, *, target: float, least: float, routes: int = 1) -> str | None:
    """Why a sweep should stop, or None to keep going.

    Two reasons, and they are different. Reaching the target is success. A round
    that adds almost nothing means looking has run out of new surfaces, and the
    answer to that is a different map rather than another pass over this one.

    The round, not the pass. A sweep is ordered route-major — every road driven
    one way before any road is driven a second — so a single pass adding nothing
    means *that road* had nothing new, which is exactly what happens when two
    routes run down the same street in opposite directions. Stopping there
    abandons every look not yet tried: measured, a sweep quit at 91.9% on a
    route that repeated itself, with six directions still to go.
    """
    if not history:
        return None
    if history[-1]["coverage"] >= target:
        return f"target reached: {history[-1]['coverage']:.1f}% >= {target:.1f}%"

    round_size = max(routes, 1)
    if len(history) <= round_size:
        return None
    gained = sum(entry["gained"] for entry in history[-round_size:])
    if gained < least:
        return (f"a round of {round_size} added {gained:.1f}%, below "
                f"{least:.1f}%; stopping at {history[-1]['coverage']:.1f}%")
    return None

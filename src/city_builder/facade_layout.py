"""What a facade looks like, drawn from the geometry we already have.

A texturing model is bad at exactly the thing a facade is made of: a regular
grid of windows sitting on evenly spaced floors. Asked in words for "six
storeys", it produces a wall with windows somewhere. That is not a failure of
the prompt, it is what low-step samplers do to periodic structure.

We do not have to ask. We *generated* these buildings, so their floor count is
not a guess — :func:`~city_builder.buildings.pick_height` snaps every height to
whole floors — and the wall is flat, so its UV layout and its front elevation
are the same picture. That is the whole idea here: rasterise the structure we
know into an image, and let the model paint materials onto it rather than
invent an architecture.

Two images come out of the same layout:

* :func:`control_image` — a line drawing for a structural conditioner
  (ControlNet canny/mlsd). This is the one that makes the windows land on the
  floors.
* :func:`procedural_facade` — a plain-pixels stand-in that needs no GPU, so the
  UV, the texel density and the whole scene-assembly path can be finished and
  checked before a model is involved at all.

Everything is expressed in UV, not in metres. The wall's V axis runs 0 at the
pavement to 1 at the roofline whatever the building's height, so a sheet
belongs to a *floor count* rather than to a height, and a sheet drawn for six
floors is only correct on a six-floor building. :func:`floor_alignment` is how
we check that a generated sheet kept the structure it was given, and
:func:`diversity` is how we check that a set of them did not all come out the
same — a score that only reads structure quietly selects for a city of
identical grey concrete.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Sheets carry their floor count in the filename: a sheet is only valid on a
# building with that many floors, and the assembly step has to be able to tell.
SHEET_PATTERN = re.compile(r"_f(\d+)(?=[_.])")


@dataclass
class FacadeLayout:
    """The structure of one facade sheet, in UV.

    The fractions are all relative — a window is a share of its bay and of its
    floor — because the sheet is stretched over the building by the UV rather
    than placed in metres. That keeps the layout exact instead of nearly right:
    the sheet spans the building precisely, whatever its height.
    """

    floors: int = 6
    bays: int = 4  # window columns across one sheet width

    # The ground floor is a shopfront, not a flat, so it is taller and its
    # openings are wider. Weighted rather than given in metres, so the floor
    # lines still add up to exactly 1.
    ground_floor_ratio: float = 1.3

    window_width: float = 0.52  # share of a bay
    window_height: float = 0.55  # share of a floor
    window_base: float = 0.25  # where the sill sits above the floor line

    shopfront_width: float = 0.80
    shopfront_height: float = 0.68
    shopfront_base: float = 0.08

    parapet: float = 0.035  # share of the total height left solid at the top

    def __post_init__(self) -> None:
        if self.floors < 1:
            raise ValueError(f"a building has at least one floor, got {self.floors}")
        if self.bays < 1:
            raise ValueError(f"a facade has at least one bay, got {self.bays}")

    # -- structure ---------------------------------------------------------

    def floor_lines(self) -> list[float]:
        """V of every floor boundary, from 0 at the pavement to 1 at the roof.

        ``floors + 1`` values: the pavement, each ceiling, and the roofline.
        """
        weights = [self.ground_floor_ratio] + [1.0] * (self.floors - 1)
        total = sum(weights)
        lines, run = [0.0], 0.0
        for weight in weights:
            run += weight
            lines.append(run / total)
        lines[-1] = 1.0  # exactly, not 0.9999999
        return lines

    def bay_lines(self) -> list[float]:
        """U of every bay boundary, 0 to 1 across one sheet.

        Evenly spaced, so U=0 and U=1 fall on the same place in the pattern and
        the sheet repeats around the building without a jump.
        """
        return [i / self.bays for i in range(self.bays + 1)]

    def windows(self) -> list[tuple[float, float, float, float]]:
        """Every opening as ``(u0, u1, v0, v1)``, the ground floor included."""
        lines = self.floor_lines()
        rects = []
        for floor in range(self.floors):
            v0, v1 = lines[floor], lines[floor + 1]
            span = v1 - v0
            ground = floor == 0
            width = self.shopfront_width if ground else self.window_width
            height = self.shopfront_height if ground else self.window_height
            base = self.shopfront_base if ground else self.window_base

            top = min(v0 + span * (base + height), 1.0 - self.parapet)  # never pierce the parapet
            bottom = v0 + span * base
            if top <= bottom:
                continue

            for bay in range(self.bays):
                centre = (bay + 0.5) / self.bays
                half = width / (2 * self.bays)
                rects.append((centre - half, centre + half, bottom, top))
        return rects

    # -- sizing ------------------------------------------------------------

    def pixel_size(self, px_per_floor: int = 128, px_per_bay: int = 128) -> tuple[int, int]:
        """``(width, height)`` at a fixed number of texels per floor and bay.

        Constant texels per floor rather than a constant sheet size: a
        three-storey sheet and a twenty-storey one otherwise end up with wildly
        different texel densities on walls standing next to each other. Both
        dimensions are rounded to a multiple of 8, which is what a latent
        diffusion VAE requires.
        """
        total_floors = self.ground_floor_ratio + (self.floors - 1)
        return (
            _round8(self.bays * px_per_bay),
            _round8(total_floors * px_per_floor),
        )

    def texel_metres(self, building_height: float, height_px: int) -> float:
        """How many metres one texel spans vertically on a real building."""
        return building_height / max(1, height_px)


def _round8(value: float) -> int:
    return max(8, round(value / 8.0) * 8)


KINDS = ("commercial", "house")


def sample_layout(floors: int, rng, *, facade_width: float = 12.0,
                  bay_metres: float | None = None,
                  kind: str = "commercial") -> FacadeLayout:
    """A layout for ``floors`` storeys, drawn at random within plausible bounds.

    One canonical drawing per floor count was the right thing while the
    mechanism was being verified, and the wrong thing for a city: it gives
    every building the same window proportions, the same bay rhythm and the
    same shopfront, and the conditioner then holds the model to it. Structure
    is the half of a facade's variety that a prompt cannot supply.

    The window width does most of the work. Narrow openings in a wide pier read
    as a punched-window block; at the top of the range the piers thin out to
    mullions and the same code draws a ribbon window.

    ``kind`` decides whether the ground floor is a shop. It has to, because
    these numbers were written for a mid-rise and applied to houses they are
    absurd: at a ground-floor ratio of 1.8 the shop takes nearly two thirds of
    a two-storey building's height, and a shopfront 0.92 of a bay wide glazes
    the whole of it. A house has the same window on every floor, a narrower bay
    — one window and a pier, not a structural span — and no shop at all.
    """
    if kind not in KINDS:
        raise ValueError(f"facade kind must be one of {KINDS}, not {kind!r}")
    if bay_metres is None:
        bay_metres = 3.0 if kind == "commercial" else 2.0
    bays = max(1, round(facade_width / rng.uniform(bay_metres * 0.8, bay_metres * 1.5)))

    if kind == "house":
        width = rng.uniform(0.30, 0.52)
        height = rng.uniform(0.32, 0.46)
        return FacadeLayout(
            floors=floors,
            bays=bays,
            ground_floor_ratio=rng.uniform(0.95, 1.15),
            window_width=width,
            window_height=height,
            window_base=rng.uniform(0.22, 0.38),
            # No shop: the ground floor gets the same opening as the rest, give
            # or take the entrance being a little taller.
            shopfront_width=width * rng.uniform(0.9, 1.35),
            shopfront_height=height * rng.uniform(1.0, 1.3),
            shopfront_base=rng.uniform(0.04, 0.14),
            parapet=rng.uniform(0.0, 0.02),
        )

    return FacadeLayout(
        floors=floors,
        bays=bays,
        ground_floor_ratio=rng.uniform(1.0, 1.8),
        window_width=rng.uniform(0.34, 0.88),
        window_height=rng.uniform(0.40, 0.70),
        window_base=rng.uniform(0.14, 0.32),
        shopfront_width=rng.uniform(0.70, 0.92),
        shopfront_height=rng.uniform(0.55, 0.78),
        parapet=rng.uniform(0.01, 0.06),
    )


def bays_for(facade_width: float, bay_metres: float = 3.0) -> int:
    """How many window columns fit in one sheet's worth of wall.

    Derived from the sheet's metric width so the windows come out the same size
    on every building, rather than being a free-floating count.
    """
    return max(1, round(facade_width / bay_metres))


# ---------------------------------------------------------------------------
# Rasterising
# ---------------------------------------------------------------------------


def _rows(v0: float, v1: float, height: int) -> tuple[int, int]:
    """Pixel row range for a V span. V=0 is the pavement, which is the *last* row."""
    top = round((1.0 - v1) * height)
    bottom = round((1.0 - v0) * height)
    return max(0, min(height, top)), max(0, min(height, bottom))


def _cols(u0: float, u1: float, width: int) -> tuple[int, int]:
    return max(0, min(width, round(u0 * width))), max(0, min(width, round(u1 * width)))


def _fill(image: np.ndarray, rect: tuple[float, float, float, float], value) -> None:
    u0, u1, v0, v1 = rect
    r0, r1 = _rows(v0, v1, image.shape[0])
    c0, c1 = _cols(u0, u1, image.shape[1])
    if r1 > r0 and c1 > c0:
        image[r0:r1, c0:c1] = value


def _outline(image: np.ndarray, rect: tuple[float, float, float, float], value, thickness: int) -> None:
    u0, u1, v0, v1 = rect
    r0, r1 = _rows(v0, v1, image.shape[0])
    c0, c1 = _cols(u0, u1, image.shape[1])
    if r1 <= r0 or c1 <= c0:
        return
    image[r0:min(r0 + thickness, r1), c0:c1] = value
    image[max(r1 - thickness, r0):r1, c0:c1] = value
    image[r0:r1, c0:min(c0 + thickness, c1)] = value
    image[r0:r1, max(c1 - thickness, c0):c1] = value


def _band(image: np.ndarray, v: float, value, thickness: int) -> None:
    """A horizontal line at V, clamped to stay inside the image."""
    row = round((1.0 - v) * image.shape[0])
    r0 = max(0, min(image.shape[0] - thickness, row - thickness // 2))
    image[r0:r0 + thickness, :] = value


def control_image(layout: FacadeLayout, width: int, height: int, *, thickness: int = 0) -> np.ndarray:
    """The layout as a line drawing, for a structural conditioner.

    White lines on black, in the shape a canny detector would produce from a
    photograph of this building: the openings, the floor line above the
    shopfront, and the parapet. Deliberately *not* a filled rendering — a
    conditioner should be told where the edges are and left to decide what the
    surfaces are made of.
    """
    thickness = thickness or max(1, round(height / 400))
    image = np.zeros((height, width, 3), dtype=np.uint8)

    lines = layout.floor_lines()
    # The fascia above the shopfront is the strongest line on a real facade,
    # and the parapet caps the composition. The intermediate floor lines are
    # left out: on a flat curtain wall there is nothing there to see, and
    # drawing them invites the model to band the whole wall.
    _band(image, lines[1], 255, thickness)
    _band(image, 1.0 - layout.parapet, 255, thickness)

    for rect in layout.windows():
        _outline(image, rect, 255, thickness)
    return image


# ---------------------------------------------------------------------------
# A stand-in that needs no model
# ---------------------------------------------------------------------------


def _periodic_noise(height: int, width: int, seed: int, roughness: float = 0.7) -> np.ndarray:
    """1/f noise that wraps on both axes, by construction rather than by luck."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    radius[0, 0] = 1.0

    spectrum = np.fft.fft2(rng.normal(size=(height, width))) / radius ** (1.0 + roughness)
    spectrum[0, 0] = 0.0
    field = np.real(np.fft.ifft2(spectrum))
    return (field - field.mean()) / (field.std() + 1e-9)


def procedural_facade(
    layout: FacadeLayout,
    width: int,
    height: int,
    *,
    seed: int = 0,
    wall=(0.62, 0.60, 0.57),
    glass=(0.16, 0.20, 0.26),
) -> np.ndarray:
    """A facade sheet built from the layout alone — no GPU, no model.

    It is not meant to look photographic. It is meant to be *correct*: the
    windows are exactly on the floors, it wraps horizontally, and it carries
    the floor rhythm :func:`floor_rhythm` measures. That makes it the reference
    a generated sheet is compared against, and it makes the rest of the
    pipeline — UV, texel density, material slots, export — finishable and
    testable on a machine with no card in it.
    """
    rng = np.random.default_rng(seed)
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:, :] = np.asarray(wall, dtype=np.float32)

    # Weathering: periodic so the sheet still wraps once it is applied.
    grain = _periodic_noise(height, width, seed)[:, :, None]
    image *= np.clip(1.0 + 0.09 * grain, 0.75, 1.25)

    lines = layout.floor_lines()
    # A spandrel band under each floor line reads as the slab edge.
    for v in lines[1:-1]:
        span = 0.012
        _fill(image, (0.0, 1.0, v - span, v), np.asarray(wall, dtype=np.float32) * 0.88)
    _fill(image, (0.0, 1.0, 1.0 - layout.parapet, 1.0), np.asarray(wall, dtype=np.float32) * 1.06)

    glass = np.asarray(glass, dtype=np.float32)
    for index, rect in enumerate(layout.windows()):
        # Every pane a little different: a wall of identical windows reads as
        # a texture, a wall of nearly-identical ones reads as a building.
        tint = glass * float(rng.uniform(0.82, 1.22))
        tint[2] *= float(rng.uniform(0.95, 1.15))  # sky reflects blue
        _fill(image, rect, np.clip(tint, 0.0, 1.0))
        _outline(image, rect, np.asarray(wall, dtype=np.float32) * 1.18,
                 max(1, round(height / 500)))

    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Did the sheet keep its structure?
# ---------------------------------------------------------------------------


def _grey(sheet: np.ndarray) -> np.ndarray:
    image = sheet.astype(np.float32)
    return image.mean(axis=-1) if image.ndim == 3 else image


def _blur(profile: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along one axis. The sigma is the tolerance."""
    sigma = max(1.0, sigma)
    radius = round(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return np.convolve(profile, kernel / kernel.sum(), mode="same")


def _match(measured: np.ndarray, expected: np.ndarray) -> float:
    """Pearson correlation of two profiles: 1 = the edges are where they belong."""
    a = measured - measured.mean()
    b = expected - expected.mean()
    denominator = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b / denominator) if denominator > 1e-12 else 0.0


def _edge_profile(image: np.ndarray, axis: int) -> np.ndarray:
    """Where an image's edges are along one axis, summed across the other.

    Window heads, sills and slab edges are all steps in brightness, so the
    gradient energy collapses a facade to the handful of lines that define it.
    """
    grey = _grey(image)
    return np.abs(np.diff(grey, axis=axis)).mean(axis=1 - axis)


def alignment(sheet: np.ndarray, control: np.ndarray, *, axis: int = 0,
              tolerance: float = 0.01) -> float:
    """How well a sheet's structure follows the control image it was given.

    A matched filter: the control image's own edges are the template, and the
    score is how strongly the sheet's edges correlate with them. ``tolerance``
    is how far an edge may drift and still count, as a fraction of the sheet's
    size along that axis. Both profiles are blurred by it — blurring only the
    template would penalise a perfect match, since a sharp edge correlates
    poorly with a smeared one.

    Deliberately *not* a measure of periodicity, which is what this started as
    and what does not work. A facade profile has roughly two impulses per storey
    — a head and a sill — so it carries a strong half-floor beat of its own, and
    "does this repeat once per floor" scores a correct sheet no better than one
    with twice the storeys. "Are the edges where we asked" has no such
    ambiguity, and it is the question that actually matters: the windows have to
    land on *this* building's floors, not merely be evenly spaced.

    1 is exactly as drawn, 0 is no relation, negative means structure precisely
    where the control said there should be none. The sheets are resampled onto
    a common length first, so a control image and a sheet of different sizes
    can still be compared.
    """
    measured = _edge_profile(sheet, axis)
    template = _edge_profile(control, axis)
    if measured.size < 4 or template.size < 4:
        return 0.0

    if template.size != measured.size:  # compare shapes, not resolutions
        source = np.linspace(0.0, 1.0, template.size)
        target = np.linspace(0.0, 1.0, measured.size)
        template = np.interp(target, source, template)

    sigma = tolerance * measured.size
    return _match(_blur(measured, sigma), _blur(template, sigma))


def floor_alignment(sheet: np.ndarray, layout: FacadeLayout, *, tolerance: float = 0.10) -> float:
    """:func:`alignment` against the layout's own drawing, down the sheet.

    ``tolerance`` is a share of one floor's height rather than of the whole
    sheet, so it means the same thing on a three-storey block as on a tower.
    """
    height, width = _grey(sheet).shape
    floors_tall = layout.ground_floor_ratio + layout.floors - 1
    return alignment(sheet, control_image(layout, width, height), axis=0,
                     tolerance=tolerance / floors_tall)


def bays_in(control: np.ndarray) -> int:
    """How many times the control image repeats across its width.

    Read off the drawing rather than passed in, so the seam measurement cannot
    drift from the layout that produced the sheet.

    Tested directly rather than read off a spectrum: each bay contributes
    several edges — two window sides, each drawn as an outline, so a doublet
    apiece — and the strongest frequency in the drawing is a high harmonic of
    the bay count, not the bay count. Rolling the image and checking it lands
    on itself has no such ambiguity.
    """
    grey = _grey(control)
    width = grey.shape[1]
    contrast = float(np.abs(grey - grey.mean()).mean())
    if contrast < 1e-9:
        return 1

    found = 1
    for bays in range(2, 17):
        shift, remainder = divmod(width, bays)
        if remainder:
            continue
        if float(np.abs(grey - np.roll(grey, shift, axis=1)).mean()) < 0.02 * contrast:
            found = bays  # keep the largest: a pattern repeating N times also
    return found        # repeats under every divisor of N


def wrap_seam(sheet: np.ndarray, control: np.ndarray) -> float:
    """The step across the wrap, against the steps it is equivalent to.

    A facade sheet repeats every bay, so its wrap falls on a bay boundary — a
    pier, a mullion, a structural line that is *supposed* to be a hard edge.
    :func:`city_builder.texture.seam_error` compares that against the mean step
    over the whole sheet, which compares a pier with blank wall: measured,
    sheets whose wrap is exactly as continuous as every other bay division
    scored anywhere from 0.3 to 11 that way, and the diagnosis cost an
    afternoon.

    The wrap's peers are the other bay boundaries. Against them, 1.0 means the
    wrap looks like every other bay division — which is the most a sheet that
    tiles can be asked for — and a real seam pushes it well above 1.
    """
    grey = _grey(sheet)
    width = grey.shape[1]
    bays = bays_in(control)

    steps = np.abs(np.diff(grey, axis=1)).mean(axis=0)
    wrap = float(np.abs(grey[:, 0] - grey[:, -1]).mean())
    peers = [steps[min(round(k * width / bays), steps.size) - 1] for k in range(1, bays)]
    if not peers:
        return wrap / (float(np.median(steps)) + 1e-9)
    return wrap / (float(np.median(peers)) + 1e-9)


def bay_alignment(sheet: np.ndarray, layout: FacadeLayout, *, tolerance: float = 0.10) -> float:
    """:func:`alignment` across the sheet. ``tolerance`` is a share of one bay."""
    height, width = _grey(sheet).shape
    return alignment(sheet, control_image(layout, width, height), axis=1,
                     tolerance=tolerance / layout.bays)


# ---------------------------------------------------------------------------
# Sheets on disk
# ---------------------------------------------------------------------------


def descriptor(sheet: np.ndarray) -> np.ndarray:
    """A short vector standing for "what this facade looks like".

    Lightness, colour on two axes, saturation and how busy the surface is —
    enough to tell a brick block from a glass tower, and nothing about where
    the windows are, which :func:`alignment` already covers.
    """
    image = _grey(sheet) if sheet.ndim == 2 else sheet.astype(np.float32)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    r, g, b = (image[..., i].mean() for i in range(3))
    high, low = image.max(axis=2), image.min(axis=2)
    return np.array([
        image.mean() / 255.0,
        (r - b) / 255.0,
        (g - (r + b) / 2) / 255.0,
        float(np.mean((high - low) / (high + 1e-9))),
        float(np.abs(np.diff(_grey(image), axis=0)).mean()) / 255.0,
    ], dtype=np.float64)


def diversity(sheets) -> float:
    """Mean distance between sheets in :func:`descriptor` space.

    The score that ranks a sheet cannot see colour, so ranking alone quietly
    selects for whatever the model does most literally — measured, a whole city
    of grey concrete at saturation 0.06, and nobody noticed until it was
    rendered. This is the other half of the measurement. Sheets from one prompt
    score about 0.05; sheets spread across the style set score about 0.4.
    """
    vectors = [descriptor(sheet) for sheet in sheets]
    if len(vectors) < 2:
        return 0.0
    total, pairs = 0.0, 0
    for i, a in enumerate(vectors):
        for b in vectors[i + 1:]:
            total += float(np.linalg.norm(a - b))
            pairs += 1
    return total / pairs


def saturation(sheet: np.ndarray) -> float:
    """How far from grey the sheet is. Concrete sits near 0.05, brick near 0.25."""
    image = sheet.astype(np.float32)
    if image.ndim == 2:
        return 0.0
    high, low = image.max(axis=2), image.min(axis=2)
    return float(np.mean((high - low) / (high + 1e-9)))


def sheet_name(floors: int, variant: int, prefix: str = "facade") -> str:
    """``facade_f06_003.png`` — the floor count has to survive the filesystem."""
    return f"{prefix}_f{floors:02d}_{variant:03d}.png"


def sheet_floors(path: str) -> int | None:
    """The floor count a sheet was drawn for, or None if it does not say."""
    match = SHEET_PATTERN.search(path.replace("\\", "/").rsplit("/", 1)[-1])
    return int(match.group(1)) if match else None


def draw_family(output_dir: str, counts: Sequence[int], *, variants: int = 4,
                facade_width: float = 24.0, bay_metres: float = 4.0,
                floor_height: float = 3.0, px_per_floor: int = 128,
                px_per_bay: int = 128, seed: int = 0,
                control: bool = True) -> dict[str, Any]:
    """One family of stand-in sheets per floor count, and the drawings behind them.

    No model and no GPU: this is the geometry half. The sheets are plain
    stand-ins that finish and check the UV path, and the control images beside
    them are what a diffusion pass is conditioned on so its windows land on the
    same storeys.

    A drawing per *variant*, not per floor count. One canonical layout would
    give every building in the city the same window proportions and the same
    bay rhythm, and the conditioner then holds the model to it — structure is
    the half of a facade's variety that no prompt supplies.
    """
    import os
    import random

    from .texture import save_tile, seam_error_axis

    counts = list(counts)
    control_dir = os.path.join(output_dir, "control")
    os.makedirs(output_dir, exist_ok=True)
    if control:
        os.makedirs(control_dir, exist_ok=True)

    sheets, seams, floors_score, bays_score, densities = [], [], [], [], []
    for floors in counts:
        for variant in range(variants):
            rng = random.Random(seed + 1000 * floors + variant)
            layout = sample_layout(floors, rng, facade_width=facade_width,
                                   bay_metres=bay_metres)
            width, height = layout.pixel_size(px_per_floor, px_per_bay)
            if control:
                save_tile(control_image(layout, width, height),
                          os.path.join(control_dir, sheet_name(floors, variant, "control")))
            sheet = procedural_facade(layout, width, height,
                                      seed=seed + 1000 * floors + variant)
            path = os.path.join(output_dir, sheet_name(floors, variant))
            save_tile(sheet, path)
            sheets.append(path)
            seams.append(seam_error_axis(sheet, axis=1))
            floors_score.append(floor_alignment(sheet, layout))
            bays_score.append(bay_alignment(sheet, layout))
            densities.append(100.0 * layout.texel_metres(floors * floor_height, height))

    return {
        "sheets": sheets,
        "control_dir": control_dir if control else None,
        "counts": counts,
        "bays": bays_for(facade_width, bay_metres),
        "seam": sum(seams) / len(seams) if seams else None,
        "floor_alignment": min(floors_score) if floors_score else None,
        "bay_alignment": min(bays_score) if bays_score else None,
        "texel_cm": [min(densities), max(densities)] if densities else None,
    }

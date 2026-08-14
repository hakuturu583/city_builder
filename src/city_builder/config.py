"""One config file for every knob, as nested dataclasses.

The options were already dataclasses, one per concern; what was missing was a
way to write them down. A kerb 0.15 m high and a parapet 1.1 m high are not
defaults so much as decisions about what kind of road this is, and decisions
belong in a file next to the map rather than in a shell history.

    surfaces:
      curb_height: 0.15
    viaduct:
      parapet_height: 1.1
      pier_spacing: 28.0

Unknown keys are an error rather than a shrug. A silently ignored typo in a
config file is the worst possible outcome: the run succeeds, the setting does
nothing, and the only evidence is in the geometry.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any

from . import ground as ground_module
from .buildings import BuildingOptions
from .extend import ExtendOptions
from .markings import MarkingOptions
from .surfaces import SurfaceOptions
from .viaduct import ViaductOptions


@dataclass
class GroundOptions:
    """The reconstructed terrain, and how the elevated set is separated from it."""

    enabled: bool = True
    cell: float = ground_module.DEFAULT_CELL  # grid spacing (m)
    smooth: float = ground_module.DEFAULT_SMOOTH  # smoothing radius, in cells
    z_gap: float = ground_module.DEFAULT_Z_GAP  # separation before two lanelets are stacked
    min_overlap: float = ground_module.DEFAULT_MIN_OVERLAP  # minimum overlap area (m2)
    clearance: float = ground_module.DEFAULT_CLEARANCE  # above the street before a ramp is elevated
    drop: float = 0.05  # hold the ground this far under the road
    fill_island: float = 0.0  # absorb junction scraps below this area (m2)


@dataclass
class CityConfig:
    """Every option group, in one object that can be read from a file."""

    surfaces: SurfaceOptions = field(default_factory=SurfaceOptions)
    extend: ExtendOptions = field(default_factory=ExtendOptions)
    ground: GroundOptions = field(default_factory=GroundOptions)
    buildings: BuildingOptions = field(default_factory=BuildingOptions)
    markings: MarkingOptions = field(default_factory=MarkingOptions)
    viaduct: ViaductOptions = field(default_factory=ViaductOptions)

    # -- reading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CityConfig:
        if not data:
            return cls()
        known = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(
                f"unknown config section(s): {', '.join(sorted(unknown))}; "
                f"expected {', '.join(sorted(known))}"
            )
        return cls(**{
            name: _section(known[name].type, name, values)
            for name, values in data.items()
        })

    @classmethod
    def from_yaml(cls, path: str) -> CityConfig:
        import yaml

        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    # -- writing -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: {g.name: getattr(getattr(self, f.name), g.name)
                     for g in fields(getattr(self, f.name))}
            for f in fields(self)
        }

    def write_yaml(self, path: str) -> str:
        import yaml

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_HEADER)
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=False)
        return path


_SECTIONS = {
    "surfaces": SurfaceOptions,
    "extend": ExtendOptions,
    "ground": GroundOptions,
    "buildings": BuildingOptions,
    "markings": MarkingOptions,
    "viaduct": ViaductOptions,
}

_HEADER = """\
# city_builder configuration. Every key is optional; what is left out keeps its
# default. `city-builder config --output city.yaml` writes this file out again.
"""


def _section(_declared, name: str, values: Any):
    """Build one option dataclass, refusing keys it does not have."""
    kind = _SECTIONS[name]
    if values is None:
        return kind()
    if not isinstance(values, dict):
        raise TypeError(f"config section {name!r} should be a mapping, got {type(values).__name__}")

    known = {f.name for f in fields(kind)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in {name!r}: {', '.join(sorted(unknown))}; "
            f"expected one of {', '.join(sorted(known))}"
        )
    return kind(**values)


def describe() -> list[tuple[str, str, str, Any]]:
    """``(section, key, type, default)`` for every option, for a help listing."""
    rows = []
    for section, kind in _SECTIONS.items():
        for f in fields(kind):
            default = f.default if f.default is not MISSING else None
            type_name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "?")
            rows.append((section, f.name, type_name, default))
    return rows


def merge_overrides(config: CityConfig, section: str, **overrides) -> CityConfig:
    """Apply non-None keyword overrides onto one section, returning a new config.

    Command-line flags beat the file — someone who typed ``--curb-height 0.2``
    means it, whatever the file says.
    """
    current = getattr(config, section)
    if not is_dataclass(current):
        raise KeyError(section)
    known = {f.name for f in fields(current)}
    applied = {k: v for k, v in overrides.items() if v is not None and k in known}
    if not applied:
        return config
    return CityConfig(**{
        f.name: (type(current)(**{**{g.name: getattr(current, g.name) for g in fields(current)},
                                  **applied})
                 if f.name == section else getattr(config, f.name))
        for f in fields(config)
    })

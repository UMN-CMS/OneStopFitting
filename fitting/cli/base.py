from __future__ import annotations
import logging
import click
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.windowing import WindowConfig

logger = logging.getLogger("fitting")


class CommaSeparatedFloat(click.ParamType):
    name = "comma_float"

    def convert(self, value, param, ctx):
        try:
            return [float(v.strip()) for v in value.split(",")]
        except AttributeError:
            self.fail(f"{value} is not a valid comma-separated list", param, ctx)


class CommaSeparatedInt(click.ParamType):
    name = "comma_int"

    def convert(self, value, param, ctx):
        try:
            return [int(v.strip()) for v in value.split(",")]
        except AttributeError:
            self.fail(f"{value} is not a valid comma-separated list", param, ctx)


def _parseWindowParams(
    window_type: str | None, window_params: tuple[str, ...]
) -> WindowConfig | None:
    from ..data.windowing import (
        GaussianWindowConfig,
        CoreDilatedWindowConfig,
        RectangularWindowConfig,
        CutWindowConfig,
    )

    if window_type is None:
        return None
    if window_type == "none":
        return None

    window_type_map = {
        "gaussian": GaussianWindowConfig,
        "core-dilated": CoreDilatedWindowConfig,
        "rectangular": RectangularWindowConfig,
        "cut": CutWindowConfig,
    }

    cls = window_type_map.get(window_type)
    if cls is None:
        valid_options = ", ".join(list(window_type_map.keys()) + ["none"])
        raise click.UsageError(f"Unknown window type '{window_type}'. Valid: {valid_options}")

    kwargs = {}
    for param in window_params:
        if "=" not in param:
            raise click.UsageError(f"Window param must be key=value, got: '{param}'")
        key, raw = param.split("=", 1)
        try:
            val = int(raw)
        except ValueError:
            try:
                val = float(raw)
            except ValueError:
                if "," in raw:
                    val = [float(x.strip()) for x in raw.split(",")]
                else:
                    val = raw
        kwargs[key] = val

    return cls(**kwargs)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """GPR Background Estimation for HEP."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

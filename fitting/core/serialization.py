from __future__ import annotations

import logging
import pickle
import json
from functools import partial
from pathlib import Path

import lz4.frame
import cattrs
import jax.numpy as jnp
import numpy as np
from cattrs.strategies import configure_tagged_union, include_subclasses
from numpyro.distributions.transforms import AffineTransform

from .data import AnalysisState
from .transforms import TransformConfig

logger = logging.getLogger(__name__)


def _makeConverter() -> cattrs.Converter:
    """Build a cattrs Converter with all custom hooks registered."""
    converter = cattrs.Converter()

    # --- JAX array hooks ---
    converter.register_unstructure_hook(jnp.ndarray, lambda v: np.asarray(v).tolist())
    converter.register_structure_hook(jnp.ndarray, lambda v, _: jnp.array(v))

    # --- numpy array hooks (for any stray numpy arrays) ---
    converter.register_unstructure_hook(np.ndarray, lambda v: v.tolist())
    converter.register_structure_hook(np.ndarray, lambda v, _: np.array(v))

    # --- Path hooks ---
    converter.register_unstructure_hook(Path, str)
    converter.register_structure_hook(Path, lambda v, _: Path(v))

    # --- AffineTransform hooks via tree_flatten / tree_unflatten ---
    def _unstructure_affine(t: AffineTransform) -> dict:
        children, aux = t.tree_flatten()
        # children = (loc, scale, domain); aux = ()
        # Serialize only loc and scale (domain is reconstructed)
        loc, scale, _domain = children
        return {
            "loc": np.asarray(loc).tolist() if hasattr(loc, "tolist") else loc,
            "scale": (
                np.asarray(scale).tolist() if hasattr(scale, "tolist") else scale
            ),
        }

    def _structure_affine(d: dict, _: type) -> AffineTransform:
        loc = jnp.array(d["loc"]) if isinstance(d["loc"], list) else d["loc"]
        scale = jnp.array(d["scale"]) if isinstance(d["scale"], list) else d["scale"]
        return AffineTransform(loc=loc, scale=scale)

    converter.register_unstructure_hook(AffineTransform, _unstructure_affine)
    converter.register_structure_hook(AffineTransform, _structure_affine)

    # --- Polymorphic hierarchies via include_subclasses ---
    _tagged_union = partial(configure_tagged_union, tag_name="_type")

    include_subclasses(TransformConfig, converter, union_strategy=_tagged_union)

    return converter


# Module-level converter instance
converter = _makeConverter()


def registerHierarchy(base_cls: type) -> None:
    """Register a new class hierarchy for polymorphic serialization.

    Call this from submodule __init__.py after defining all subclasses.
    """
    _tagged_union = partial(configure_tagged_union, tag_name="_type")
    include_subclasses(base_cls, converter, union_strategy=_tagged_union)


def save(state: AnalysisState, path: Path) -> None:
    """Save an AnalysisState to a compressed pickle file.

    Args:
        path: Output directory path. Will be created if it doesn't exist.
    """
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the full state as compressed pickle
    pkl_path = out_dir / "state.pklz4"
    with lz4.frame.open(pkl_path, "wb") as f:
        pickle.dump(state, f)

    logger.info(f"Saved analysis state to {pkl_path}")

    # Save summary as JSON
    summary = {
        "config": converter.unstructure(state.config),
        "metadata": converter.unstructure(state.metadata),
    }

    if state.diagnostic_metrics is not None:
        summary["metrics"] = converter.unstructure(state.diagnostic_metrics)

    if state.training_result is not None:
        summary["training"] = {
            "final_loss": float(state.training_result.final_loss),
            "metric_histories": converter.unstructure(
                state.training_result.metric_histories
            ),
        }

    if state.ppc_results is not None:
        summary["ppc"] = {
            "test_stats": converter.unstructure(state.ppc_results["test_stats"]),
        }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved summary to {json_path}")


def load(path: Path) -> AnalysisState:
    """Load an AnalysisState from a compressed pickle file.

    Args:
        path: Directory containing 'state.pklz4' or direct path to a file.
    """
    p = Path(path)
    if p.is_dir():
        pkl_path = p / "state.pklz4"
    else:
        pkl_path = p

    if not pkl_path.exists():
        raise FileNotFoundError(f"State file {pkl_path} does not exist.")

    with lz4.frame.open(pkl_path, "rb") as f:
        state = pickle.load(f)

    logger.info(f"Loaded analysis state from {pkl_path}")
    return state

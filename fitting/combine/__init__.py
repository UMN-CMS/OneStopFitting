"""Higgs Combine output generation."""

from ..core.serialization import registerHierarchy
from .commands import CombineCommand

registerHierarchy(CombineCommand)

"""Data loading, windowing, and preprocessing."""

from ..core.serialization import registerHierarchy
from .loading import FileLoader
from .windowing import Window

# Register hierarchies for polymorphic serialization
registerHierarchy(Window)
registerHierarchy(FileLoader)

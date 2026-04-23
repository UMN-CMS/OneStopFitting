"""Data loading, windowing, and preprocessing."""

from ..core.serialization import registerHierarchy
from .loading import FileLoader
from .windowing import Window, WindowConfig

# Register hierarchies for polymorphic serialization
registerHierarchy(Window)
registerHierarchy(WindowConfig)
registerHierarchy(FileLoader)


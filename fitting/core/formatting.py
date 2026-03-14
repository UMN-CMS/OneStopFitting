"""Metadata format string utilities.

Provides dot-notation format strings from nested metadata dicts,
enabling metadata-driven plot titles, output paths, bookkeeping
labels, etc.

Example::

    metadata = {
        'era': {'name': '2018', 'lumi': 59.83},
        'other_data': {'stop_mass': 1000, 'chargino_mass': 400},
    }
    template = "{era.name} - mStop={other_data.stop_mass} GeV"
    result = dotFormat(template, **dict(dictToDot(metadata)))
    # => "2018 - mStop=1000 GeV"
"""

from __future__ import annotations

import string
from collections.abc import Iterator, Mapping
from typing import Any


def dictToDot(dictionary: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    """Flatten a nested dict to dot-separated key/value pairs.

    Args:
        dictionary: Potentially nested mapping.

    Yields:
        Tuples of (dot.separated.key, leaf_value).

    Example::

        >>> list(dictToDot({"a": {"b": 1, "c": {"d": 2}}}))
        [("a.b", 1), ("a.c.d", 2)]
    """
    for field, value in dictionary.items():
        if isinstance(value, Mapping):
            for key, val in dictToDot(value):
                yield f"{field}.{key}", val
        else:
            yield field, value


def dotFormat(template: str, **kwargs: Any) -> str:
    """Format a string template using dot-notation keyword arguments.

    Unlike str.format(), this does not choke on dotted field names
    — it resolves them from the flat kwargs produced by dictToDot().

    Args:
        template: A format string with ``{field.subfield}`` placeholders.
        **kwargs: Flattened dot-notation key/value pairs.

    Returns:
        The formatted string.

    Raises:
        KeyError: If a placeholder has no matching kwarg.

    Example::

        >>> dotFormat("{era.name} L={era.lumi} fb-1", **{"era.name": "2018", "era.lumi": 59.83})
        '2018 L=59.83 fb-1'
    """
    parsed = string.Formatter().parse(template)
    result = ""
    for literal, field_name, _, _ in parsed:
        result += literal
        if field_name is not None:
            result += str(kwargs[field_name])
    return result


def formatFromMetadata(template: str, metadata: Mapping[str, Any]) -> str:
    """Convenience: flatten metadata and format in one call.

    Args:
        template: Format string with ``{dot.separated}`` placeholders.
        metadata: Nested metadata dict.

    Returns:
        Formatted string.
    """
    return dotFormat(template, **dict(dictToDot(metadata)))

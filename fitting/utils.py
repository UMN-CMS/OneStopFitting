import itertools as it
from collections.abc import Mapping, MutableMapping, Generator
import re
from collections import OrderedDict
from attrs import asdict
from pathlib import Path
import string
from typing import Any, TypeVar
import copy

T = TypeVar("T")


def dictToDot(dictionary: Mapping[str, T]) -> Generator[tuple[str, T]]:
    for field, value in dictionary.items():
        if isinstance(value, Mapping):
            for key, val in dictToDot(value):
                yield f"{field}.{key}", val
        else:
            yield field, value


def dotFormat(template: str, **kwargs) -> str:
    parsed = string.Formatter().parse(template)
    result = ""
    for literal, field_name, _, _ in parsed:
        result += literal
        if field_name is not None:
            result += str(kwargs[field_name])
    return result


def formatFromMetadata(template: str, metadata: Mapping[str, Any]) -> str:
    return dotFormat(template, **dict(dictToDot(metadata)))


def merge(destination: MutableMapping, source: Mapping) -> Mapping:
    for key, value in source.items():
        if isinstance(value, Mapping):
            node = destination.setdefault(key, {})
            merge(node, value)
        else:
            destination[key] = value
    return destination


def dotToNested(dot_dict: Mapping) -> Mapping:
    nested = {}
    for key, value in dot_dict.items():
        keys = key.split(".")
        node = nested
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return nested


def iterComboDict(**kwargs):
    okwargs = OrderedDict(kwargs)
    for combo in it.product(*okwargs.values()):
        yield dict(zip(okwargs.keys(), combo))


def iterComboDictNested(**kwargs):
    okwargs = OrderedDict(kwargs)
    for combo in it.product(*okwargs.values()):
        yield dotToNested(dict(zip(okwargs.keys(), combo)))


def evolveCombos(obj, **kwargs):
    okwargs = OrderedDict(kwargs)
    for combo in it.product(*okwargs.values()):
        if isinstance(obj, Mapping):
            base = copy.deepcopy(dict(obj))
        else:
            base = asdict(obj)

        updated = merge(base, dotToNested(dict(zip(okwargs.keys(), combo))))

        if isinstance(obj, Mapping):
            yield updated
        else:
            yield type(obj)(updated)


def getSignal(path):
    match = re.search(r"signal_.+_(31\d)_(\d+)_(\d+)", str(path))
    return [match.group(1), int(match.group(2)), int(match.group(3))]


def getRecoCategory(name):
    n = str(Path(name).name)
    found = next(
        (x for x in ["uncomp_", "verycomp_", "comp_"] if n.startswith(x)), None
    )
    return found
    # if not found:
    #     raise RuntimeError("Could not determine reco category.")
    # return found


# def getCategory(mstop, mchi):
#     if mchi < (0.6 * mstop + 150):
#         return "uncomp"
#     if mchi < (0.75 * mstop + 150):
#         return "comp"
#     else:
#         return "verycomp"


def commonDict(items):
    i = iter(items)
    ret = copy.deepcopy(dict(next(i)))
    for item in i:
        data = item
        for k in data:
            if k in ret and not ret[k] == data[k]:
                del ret[k]
    return ret


def getCategory(mstop, mchi):
    if mchi < (0.75 * mstop):
        return "uncomp"
    if mchi < (0.90 * mstop):
        return "comp"
    else:
        return "verycomp"

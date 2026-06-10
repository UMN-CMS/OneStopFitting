from jax.numpy import isin
import fnmatch
import itertools as it
from collections.abc import Mapping, MutableMapping, Generator
import re
from collections import OrderedDict, defaultdict
import attrs
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


def getCategory(mstop, mchi):
    if mchi < (0.75 * mstop):
        return "uncomp"
    if mchi < (0.90 * mstop):
        return "comp"
    else:
        return "verycomp"


def commonDict(items):
    i = iter(items)
    ret = copy.deepcopy(dict(next(i)))
    for item in i:
        data = item
        for k in data:
            if k in ret and not ret[k] == data[k]:
                del ret[k]
    return ret


def formatLines(elems: list[list[str]], separator: str = "  ") -> list[str]:
    if not elems:
        return []
    max_row_len = max(len(row) for row in elems)
    padded_elems = [row + [""] * (max_row_len - len(row)) for row in elems]
    max_lens = [max(len(str(x)) for x in col) for col in zip(*padded_elems)]
    row_format = separator.join(f"{{: <{width}}}" for width in max_lens)
    return [row_format.format(*e).rstrip() for e in padded_elems]


def formatToRegex(format_str) -> re.Pattern:
    formatter = string.Formatter()
    regex_parts = []
    used_names = set()
    for literal_text, field_name, format_spec, conversion in formatter.parse(
        format_str
    ):
        if literal_text:
            regex_parts.append(re.escape(literal_text))
        if field_name is not None:
            if field_name.isidentifier() and field_name not in used_names:
                used_names.add(field_name)
                regex_parts.append(rf"(?P<{field_name}>.*?)")
            else:
                regex_parts.append(r"(.*?)")
    pattern = "^" + "".join(regex_parts) + "$"
    return re.compile(pattern)


def formatToGlob(format_str) -> str:
    formatter = string.Formatter()
    glob_parts = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(
        format_str
    ):
        if literal_text:
            glob_parts.append(literal_text)
        if field_name is not None:
            glob_parts.append("*")
    return "".join(glob_parts)


def makeMatcherFromFormat(format_str):
    regex = formatToRegex(format_str)

    def func(text):
        return regex.match(text).groupdict()

    return func


def getSignals(signal_pattern: str, limit_correct_cat=True) -> dict[str, dict]:
    signal_glob = formatToGlob(signal_pattern)
    sig_matcher = makeMatcherFromFormat(signal_pattern)
    signals = {}
    for path in glob.glob(signal_glob, recursive=False):
        sig_params = getSignal(path)
        target_category = getCategory(sig_params[1], sig_params[2])
        sig_data = sig_matcher(str(path))
        if target_category == sig_data["category"] or not limit_correct_cat:
            signals[path] = sig_data | {
                "coupling": sig_params[0],
                "mstop": sig_params[1],
                "mchi": sig_params[2],
            }

    return signals


def getBackgrounds(background_pattern) -> dict[str, dict]:
    background_glob = formatToGlob(background_pattern)
    background_matcher = makeMatcherFromFormat(background_pattern)
    backgrounds = {}
    for path in glob.glob(background_glob, recursive=False):
        backgrounds[path] = background_matcher(str(path))
    return backgrounds


def getRuleCode(d1, d2, rules):
    ret = []
    for name, pats in rules:
        for i, (p1, p2) in enumerate(pats):
            if fnmatch.fnmatch(p1, d1[name]) and fnmatch.fnmatch(p1, d2[name]):
                ret.append((name, i))
    return ret


def getMatchingRules(d1, rules, right=True):
    return tuple(
        frozenset(
            i for i, x in enumerate(pats) if fnmatch.fnmatch(d1[k], x[int(right)])
        )
        for k, pats in rules.items()
    )


def isMatch(m1, m2):
    return all(not m1[i].isdisjoint(m2[i]) for i in range(len(m1)))

MatchRules = dict[str, list[tuple[str, str]]]

def leftGroup(
    left: dict[str, dict[str, int | float | str]],
    right: dict[str, dict[str, int | float | str]],
    match_rules: MatchRules,
):
    matches_left = {
        path: getMatchingRules(data, match_rules, False) for path, data in left.items()
    }
    matches_right = {
        path: getMatchingRules(data, match_rules) for path, data in right.items()
    }
    ret = []
    for path, k in matches_left.items():
        sig_groups: defaultdict[str, list[str]] = defaultdict(list)

        for p, m in matches_right.items():
            if not isMatch(k, m):
                continue
            d = right[p]
            sig_groups[(d["coupling"], d["mstop"], d["mchi"])].append(p)
        for g in sig_groups.values():
            ret.append((path, g))
    return ret

    # for x,matches in matches_left:


#
#     # jobs.append(
#     #     {
#     #         "signals": sig_file,
#     #         "transfer_signals": sig_file,
#     #         "background": bkg_file,
#     #         "output_dir": output_dir,
#     #         "category": category,
#     #         "mstop": sig_params[1],
#     #         "mchi": sig_params[2],
#     #         "year": year,
#     #         "config": config_pattern.format(
#     #             pipeline=pipeline,
#     #             category=category,
#     #             reco_category=reco_category,
#     #         ),
#     #         "pipeline": pipeline,
#     #         "reco_category": reco_category,
#     #         "coupling": pipeline.removeprefix("Signal"),
#     #     }
#     # )
#
#
if __name__ == "__main__":
    group_fields = ["pipeline", "year", "toy_index"]
    match_rules = {
        "year": [
            ("2018", "201*"),
            ("2017", "2017"),
            ("2016", "2016*"),
            ("Run3", "202*"),
        ],
        "category": [("comp", "comp"), ("uncomp", "uncomp"), ("verycomp", "verycomp")],
    }
    import glob
    from rich import print

    s1 = "smoothed_combined/{year}/{pipeline}/qcd_inclusive_2018/{category}/{category}_{toy_index}.pklz4"
    s2 = "export_combined/{year}/{pipeline}/signal_{}/{category}_{}.pklz4"
    g1 = formatToGlob(s1)
    b = getBackgrounds(s1)
    s = getSignals(s2)
    print(s)

    r = leftGroup(b, s, match_rules)
    print(r)
    # print(r)
    # groups = buildGroups(
    #     None, b, ["category", "toy_index", "pipeline", ("year", ("201.*", "201.*"))]
    # )
    # print(b)
    # print(s)

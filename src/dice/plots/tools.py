import pandas as pd
import fnmatch
import copy

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterator, TypeVar

T = TypeVar("T", bound="Label")


@dataclass
class Label(Generic[T]):
    name: str  # actual name of the label
    label: str  # label combined
    path: list[str]  # the label splitted
    count: int  # count of rows with this label
    group: str  # group name, i.e., the column
    children: list[T] = field(default_factory=list)  # children

    def add(self, *lab: T):
        self.children.extend(lab)


class PrepLab:
    def __init__(
        self, index: int, group: str, name: str, label: str, text: str, color: str
    ) -> None:
        self.index = index
        self.group = group
        self.name = name
        self.label = label
        self.text = text
        self.color = color


type ConfFilter = Callable[[list[PrepLab]], Iterator[PrepLab]]


@dataclass
class LabConf:
    # the label is in the format from:to:*
    # stars represent "whatever from here", kind of a glob
    label: str | None
    # the color to set
    color: str
    border: Any
    # whether to show this label
    show: bool
    # a lambda function to filter dataframes instead
    filter: ConfFilter | None = None

    def do(self, labs: list[PrepLab]) -> None:
        matches: set[PrepLab] = set()
        if self.filter:
            matches = set(self.filter(labs))

        if self.label:
            for lab in labs:
                if fnmatch.fnmatch(lab.label, self.label):
                    matches.add(lab)

        for m in matches:
            m.color = self.color


@dataclass
class LabelCollection:
    color: str = "black"
    _prep: list[PrepLab] = field(default_factory=lambda: [])

    def labels(self) -> list[str]:
        return [l.text for l in self._prep]

    def add(self, *labs: Label) -> "LabelCollection":
        def walk(lab: Label):
            for ch in sorted(lab.children, key=lambda l: l.name):
                walk(ch)
            prep = PrepLab(
                len(self._prep), lab.group, lab.name, lab.label, lab.name, self.color
            )
            self._prep.append(prep)

        for lab in labs:
            walk(lab)
        return self

    def update(self, *confs: LabConf) -> "LabelCollection":
        for conf in confs:
            # changes the drawing properties of prepared labels
            # it filters the labels and applies the changes
            conf.do(self._prep)
        return self

    def find(self, lab: str) -> PrepLab:
        for p in self._prep:
            if p.label == lab:
                return p
        raise ValueError(f"label {lab} not found")

    def all(self) -> list[PrepLab]:
        return self._prep


@dataclass
class Links:
    source: list[int] = field(default_factory=lambda: [])
    target: list[int] = field(default_factory=lambda: [])
    value: list[int] = field(default_factory=lambda: [])
    color: list[str] = field(default_factory=lambda: [])

    def from_roots(self, roots: list[Label], col: LabelCollection) -> dict:
        def walk(labs: list[Label], p_lab: str):
            p = col.find(p_lab)
            for lab in labs:
                walk(lab.children, lab.label)
                c = col.find(lab.label)
                self.add(p.index, c.index, lab.count, c.color)

        for root in roots:
            walk(root.children, root.label)

        return self.to_dict()

    def add(self, source: int, target: int, value: int, color: str) -> None:
        self.source.append(source)
        self.target.append(target)
        self.value.append(value)
        self.color.append(color)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "value": self.value,
            "color": self.color,
        }


@dataclass
class Nodes:
    ids: list[str]
    links: dict
    labels: list[str]


def filter_labels(labels: list[PrepLab], pattern: str) -> list[PrepLab]:
    return [lab for lab in labels if fnmatch.fnmatch(lab.label, pattern)]


def make_lab_conf(
    label: str | None = None,
    color: Any = "black",
    border: str = "black",
    show: bool = True,
    cfilter: ConfFilter | None = None,
) -> LabConf:
    return LabConf(label, color, border, show, cfilter)


def make_collection(
    labs: list[Label], confs: list[LabConf], color: str = "black"
) -> LabelCollection:
    return LabelCollection(color=color).add(*labs).update(*confs)


def make_labels(
    data: pd.DataFrame,
    *column: str,
    lconfs: list[LabConf] = [],
    prefix: str = "",
    color: str = "black",
) -> tuple[list[Label], LabelCollection]:
    def walk(w: pd.DataFrame, cols: list[str], path: list[str] = []) -> list[Label]:
        labs = []
        if not cols:
            return []

        c_cols = copy.copy(cols)
        group = prefix + c_cols.pop(0)

        for g_name, g_df in w.groupby(group):
            assert isinstance(g_name, str)

            name = g_name.strip('"')
            p = [*path, name]
            lab = Label(
                name=name,
                path=p,
                label=":".join(p),
                count=len(g_df.index),
                group=group,
                children=walk(g_df, c_cols, p),
            )

            labs.append(lab)
        return labs

    roots = walk(data, [*column])
    collection = make_collection(roots, lconfs, color)
    return (roots, collection)


def make_links(roots: list[Label], col: LabelCollection) -> dict:
    return Links().from_roots(roots, col)


def make_nodes(
    data: pd.DataFrame,
    *column: str,
    ids: list[str],
    lconfs: list[LabConf] = [],
    prefix: str = "",
) -> Nodes:
    roots, coll = make_labels(data, *column, lconfs=lconfs, prefix=prefix)
    links = make_links(roots, coll)

    return Nodes(
        ids=ids,
        links=links,
        labels=coll.labels(),
    )

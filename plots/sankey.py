import plotly.graph_objects as go
import pandas as pd
import fnmatch
import copy

from typing import Any
from dataclasses import dataclass, field

from dice import repo

@dataclass
class Label:
    # actual name of the label
    name: str
    # label combined
    label: str
    # the label splitted
    path: list[str]
    # count of rows with this label
    count: int
    # group name, i.e., the column
    group: str

    # children, this is a tree
    children: list['Label']
    
    def add(self, lab: 'Label'):
        self.children.append(lab)

@dataclass
class LabConf:
    # the label is in the format from:to:*
    # stars represent "whatever from here", kind of a glob
    label: str
    # the color to set
    color: Any
    # whether to show this label
    show: bool

@dataclass
class PrepLab:
    index: int
    name: str
    label: str
    text: str
    color: str

def filter_labels(labels: list[PrepLab], pattern: str) -> list[PrepLab]:
    return [lab for lab in labels if fnmatch.fnmatch(lab.label, pattern)]

@dataclass
class LabelCollection:
    _prep: list[PrepLab] = field(default_factory=lambda: [])

    def labels(self) -> list[str]:
        return [l.text for l in self._prep]
    
    def add(self, *labs: Label) -> 'LabelCollection':
        def walk(lab: Label):
            for ch in lab.children:
                walk(ch)
            prep = PrepLab(len(self._prep), lab.name, lab.label, lab.name, "black")
            self._prep.append(prep)
        for lab in labs:
            walk(lab)
        return self
    
    def update(self, *confs: LabConf) -> 'LabelCollection':
        for conf in confs:
            for p in filter_labels(self._prep, conf.label):
                if not conf.show:
                    p.text = ""
                p.color = conf.color
        return self

    def find(self, lab: str) -> PrepLab:
        for p in self._prep:
            if p.label == lab:
                return p
        raise ValueError(f"label {lab} not found")

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
            "color": self.color
        }

@dataclass
class Nodes:
    ids: list[str]
    links: dict
    labels: list[str]

class Sankey:
    def __init__(self, r: repo.Repository, *columns: str, prefix: str) -> None:
        self.columns = columns
        self.prefix = prefix
        self.repo = r

    def _links(self, roots: list[Label], col: LabelCollection) -> dict:
        return Links().from_roots(roots, col)

    def _labels(self, df: pd.DataFrame, *confs: LabConf) -> tuple[list[Label], LabelCollection]:
        "makes label groups"

        def walk(w: pd.DataFrame, cols: list[str], path: list[str]=[]) -> list[Label]:
            labs = []
            if not cols:
                return []
            
            c_cols = copy.copy(cols)
            group = "_".join([self.prefix, c_cols.pop(0)])

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
                    children=walk(g_df, c_cols, p)
                )

                labs.append(lab)
            return labs
        roots = walk(df, list(self.columns))
        collection = LabelCollection().add(*roots).update(*confs)
        return (roots,collection)
            
    
    def build(self, ids: list[str], *confs: LabConf) -> Nodes:
        # assumes the repo is already prepared with the right filters
        data = self.repo.collect()
        roots, col = self._labels(data, *confs)
        links = self._links(roots, col)

        return Nodes(
            ids = ids,
            links = links,
            labels = col.labels(),
        )
    
    def figure(self, ids: list[str], *lconf: LabConf) -> go.Figure:
        n = self.build(ids, *lconf)
        return go.Figure(go.Sankey(
            node=dict(
                thickness=10,
                label=n.labels,
                color="black",
            ),
            link=n.links
        ))

def make_sankey(r: repo.Repository, *columns: str, prefix: str = "data") -> Sankey:
    '''
    Prepares a sankey object from the data in a repository
    '''
    sankey = Sankey(r, *columns, prefix=prefix)
    return sankey


def sankey_figure(r: repo.Repository, *columns: str, ids: list[str] = [], lconfs: list[LabConf] = [], prefix: str = "data") -> go.Figure:
    '''
    Makes a simple sankey plot using the given columns and their order, from left to right.
    Multiple DFs are colored differently.
    '''
    sankey = make_sankey(r, *columns, prefix=prefix)
    return sankey.figure(ids, *lconfs)

def make_lab_conf(label: str, color: Any, show: bool = True) -> LabConf:
    return LabConf(label, color, show)

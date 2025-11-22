import matplotlib.pyplot as plt
import pandas as pd
import squarify

from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dataclasses import dataclass, field
from .tools import Label

# Quadtile
# - the biggest in the middle
# - it shrinks with droping the size
# - the parent graph is a square
# -- this shows the largest, and how it decreases
# - multi-level: e.g., country, asn, etc.

@dataclass
class Patch(Label["Patch"]):
    rect: dict = field(default_factory=dict)

def make_patches(
    df: pd.DataFrame, 
    cols: list[str],
    rect: dict,
    path: list[str] = [],
    pad: float = 0.01,
    txt_pad: float = 0.08,
    prefix: str="",
    ) -> list[Patch]:

    if not cols:
        return []
    
    col = prefix+cols[0]
    groups = (
        df.groupby(col)
         .size()
         .reset_index(name="count")
         .sort_values("count", ascending=False)
    )

    sizes = groups["count"].values
    labels = groups[col].values

    
    has_children = len(cols) > 1 and len(df) > 0

    # add padding
    txt_rsv = rect["dy"] * txt_pad


    # scale padding relative to rectangle size
    pad_x = rect["dx"] * pad
    pad_y = rect["dy"] * pad

    inner_x = rect["x"] + pad_x
    inner_y = rect["y"] + pad_y
    inner_dx = max(1e-6, rect["dx"] - 2 * pad_x)
    inner_dy = max(1e-6, rect["dy"] - 2 * pad_y - txt_rsv)

    sizes = squarify.normalize_sizes(sizes, inner_dx, inner_dy)
    rects = squarify.padded_squarify(sizes, inner_x, inner_y, inner_dx, inner_dy)

    patches: list[Patch] = []
    for r, label, size in zip(rects, labels, sizes):
        label = label.strip('"')
        p = [*path, label]
        patch = make_patch(label, "/".join(path), int(size), col, r)

        ch = make_patches(df, cols[1:], r, p, pad, txt_pad, prefix)
        patch.add(*ch)
        patches.append(patch)

    return patches

def make_patch(label, path, val, group, rect):
    return Patch(
        name=label,
        label="/".join(path),
        path=path,
        count=val,
        group=group,
        rect=rect
    )

def draw_patch(patch: Patch, ax: Axes) -> None:

    # add the rectangle
    ax.add_patch(Rectangle(
        (patch.rect["x"], patch.rect["y"]),
        patch.rect["dx"],
        patch.rect["dy"],
        edgecolor="black",
    ))

    # try to add the text in the center
    if patch.rect["dx"] > 2 and patch.rect["dy"] > 2: 
        ax.text(
            patch.rect["x"] + 1,
            patch.rect["y"] + (patch.rect["dy"]-1),
            patch.name,
            va="top", ha="left", fontsize=7
        )

    for child in patch.children:
        draw_patch(child, ax)

def mesh_figure(
    df: pd.DataFrame, 
    *columns: str,
    prefix: str = "",
    ax: Axes | None = None,
    ) -> Figure | None:

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
        ax.axis("off")

    rect = {"dx": 100, "dy": 100, "x": 0, "y": 0}
    patches = make_patches(df, list(columns), rect, [], prefix=prefix)

    for patch in patches:
        draw_patch(patch, ax)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    return fig

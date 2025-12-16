import pandas as pd
import matplotlib.pyplot as plt
import ipaddress
import numpy as np

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from netgraph import Graph
from hilbertcurve.hilbertcurve import HilbertCurve
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

from .tools import LabConf,  make_labels, make_nodes

class Network:
    def __init__(self, data: pd.DataFrame, *columns: str, prefix: str) -> None:
        self.columns = columns
        self.prefix = prefix
        self.data = data

    def figure(self, ids: list[str], *lconf: LabConf, ax: Axes) :
        n = make_nodes(self.data, *self.columns, ids=ids, lconfs=[*lconf], prefix=self.prefix)
        edges = [(f, t) for f, t in zip(n.links["source"], n.links["target"])]
        #node_edge_color = {node: color for node, color in zip(edges, n.links["color"])}

        return Graph(edges,
            #edge_alpha=.5,
            #edge_layout='bundled',
            edge_width=.1,
            node_size=1,
            node_edge_width=0.1,
            #node_layout='radial',
            #node_edge_color=node_edge_color,
            #node_layout_kwards=...,
            node_labels={idx: lab for idx, lab in enumerate(n.labels)},
            ax=ax,
        )
    
def build_network(data: pd.DataFrame, *columns: str, prefix: str = "") -> Network:
    return Network(data, *columns, prefix=prefix)

def network_figure(
        data: pd.DataFrame, 
        *columns: str, 
        ids: list[str] = [],
        lconfs: list[LabConf] = [],
        prefix: str = "",
        ax: Axes | None = None
    ) -> Figure | None: 
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
        ax.axis("off")

    # There is a sankey-like relationship that I can borrow. We only need to build "from" and "to"
    # as the links.
    # normally, we would have "asn_country", "asn_number", and the columns for the octets
    # (because I do not have the prefixes - I should collect this tho)
    net = build_network(data, *columns, prefix=prefix)
    net.figure(ids, *lconfs, ax=ax)
    return fig

def ip_to_series(ip: str, prefix: str = "ip_octet_") -> pd.Series:
    # there are 4 levels, one for each of the octets
    ip_arr = ip.split(".")
    if len(ip_arr) < 2:
        raise ValueError(f"invalid ip {ip}")
    
    if len(ip_arr) < 3:
        suffix = ip_arr.pop()
        ip_arr += [0] * (3-len(ip_arr))
        ip_arr.append(suffix)

    return pd.Series({f"{prefix}{ind}": int(val) for ind, val in enumerate(ip_arr)})    

def prefix_density(prefix: str, count: int) -> tuple[int, float]:
    """Returns the density of an IPv4 prefix, i.e., how many addresses are allocated in the available space"""
    net = ipaddress.ip_network(prefix, strict=False)
    return (net.num_addresses, count / net.num_addresses)


def draw_first_octet_grid(ax: Axes, hilbert: "HilbertCurve"):
    octet_size = 2**24  # first octet

    ax.figure.canvas.draw()  # ensure limits exist
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    diag = np.hypot(x1 - x0, y1 - y0)
    # tuning factor: larger = larger fontsize
    fontsize_base = diag * 0.00002  

    for octet in range(256):
        start_dist = octet * octet_size
        end_dist = (octet + 1) * octet_size - 1

        # sample points along this octet
        # For speed, take just a few points per octet
        samples = 10
        dists = [start_dist + i*(end_dist-start_dist)//(samples-1) for i in range(samples)]
        points = [hilbert.point_from_distance(d) for d in dists]
        xs, ys = zip(*points)

        # bounding box
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # draw rectangle
        rect = Rectangle(
            (x_min, y_min), 
            x_max-x_min, 
            y_max-y_min, 
            linestyle=(0, (5, 10)),
            edgecolor="gray", 
            lw=0.5, 
            facecolor="none", 
            alpha=0.5
        )

        ax.add_patch(rect)

        # annotate octet in top-right
        txt = ax.text(
            x_max, 
            y_max, 
            str(octet), 
            color="gray", 
            fontsize=fontsize_base,
            fontweight="bold",
            ha="right", 
            va="top", 
            alpha=0.15
        )

        txt.set_path_effects([
            path_effects.Stroke(linewidth=1.5, foreground='gray', alpha=.15),
            path_effects.Normal()
        ])

    ax.set_axis_off()
    ax.set_aspect("equal")

def plot_ips_hilbert(
        df: pd.DataFrame, 
        col: str="host",
        groups: list[str]=[], # columns to group by for the labels, e.g.: ["a", "b"] -> "a_val:b_val"
        lconfs: list[LabConf] = [],
        color: str | None = "black", # default color
        ax: Axes | None = None,
    ) -> Figure | None:
    """IP addresses mapped to 2D Hilbert curve"""
    _, collection = make_labels(df, *groups, lconfs=lconfs, color=color)

    # Step 2 — assign a color to each row
    # map label string -> PrepLab.color
    label_to_color = {pl.label: pl.color for pl in collection.all()}

    def row_label(row):
        if not groups:
            return "all"
        return ":".join(str(row[g]) for g in groups)
    df_labels = df.apply(row_label, axis=1)
    df_colors = df_labels.map(lambda lab: label_to_color.get(lab, color))
    
    # Setup Hilbert curve: 16 iterations = 2^16 x 2^16 grid = 32 bits
    hilbert = HilbertCurve(p=16, n=2)

    def ip_to_point(ip):
        val = int(ipaddress.ip_address(ip))
        return hilbert.point_from_distance(val)

    points = df[col].apply(ip_to_point) # type: ignore
    xs, ys = zip(*points)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))

    ax.scatter(xs, ys, marker="s", c=df_colors, s=.1, alpha=.5)
    draw_first_octet_grid(ax, hilbert)

    return fig
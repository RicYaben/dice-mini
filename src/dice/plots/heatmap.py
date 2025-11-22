import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.colors import ListedColormap

def heatmap_figure(df: pd.DataFrame, x:str, y: str, ax: Axes | None =None) -> Figure | SubFigure:
    data = pd.crosstab(df[x], df[y])
    mask_zero = data == 0
    mask_nonzero = ~mask_zero

    if not ax:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.axis("off")
    else:
        fig = ax.figure

    sns.heatmap(
        data,
        annot=data.where(mask_nonzero),  # only annotate non-zero values
        fmt=".0f",
        cmap="coolwarm",
        cbar=False,
        linewidths=0.5,
        linecolor='white',
        mask=mask_zero,  # mask zero cells
        ax=ax
    )

    sns.heatmap(
        data,
        annot=False,  # don't annotate zeros
        cbar=False,
        linewidths=0.5,
        linecolor='white',
        mask=mask_nonzero,  # mask non-zero cells
        cmap=ListedColormap(["#E6E6E6"]),  # light gray
        ax=ax
    )

    return fig
    

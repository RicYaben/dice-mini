import plotly.graph_objects as go
import pandas as pd

from .tools import LabConf, make_nodes

class Sankey:
    def __init__(self, data: pd.DataFrame, *columns: str, prefix: str) -> None:
        self.columns = columns
        self.prefix = prefix
        self.data = data
    
    def figure(self, ids: list[str], *lconf: LabConf) -> go.Figure:
        n = make_nodes(self.data, *self.columns, ids=ids, lconfs=[*lconf], prefix=self.prefix)
        return go.Figure(go.Sankey(
            node=dict(
                thickness=10,
                label=n.labels,
                color="black",
            ),
            link=n.links
        ))

def build_sankey(data: pd.DataFrame, *columns: str, prefix: str = "data_") -> Sankey:
    '''
    Prepares a sankey object from the data in a repository
    '''
    sankey = Sankey(data, *columns, prefix=prefix)
    return sankey


def sankey_figure(data: pd.DataFrame, *columns: str, ids: list[str] = [], lconfs: list[LabConf] = [], prefix: str = "data_") -> go.Figure:
    '''
    Makes a simple sankey plot using the given columns and their order, from left to right.
    Multiple DFs are colored differently.
    '''
    sankey = build_sankey(data, *columns, prefix=prefix)
    return sankey.figure(ids, *lconfs)
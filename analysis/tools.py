import pandas as pd

def count_groups(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    "returns a grouped dataframe with counts using a list of columns. The order of the columns determines how the groups are formed"
    
    # Group by the specified columns and count the occurrences
    grouped = df.groupby(cols).size().reset_index(name='count')
    
    # Now we want to create a MultiIndex with two levels: the original columns and the count
    grouped.set_index(cols, inplace=True)  # Set both cols as MultiIndex
    grouped = grouped.sort_index()  # Sort by index to keep order

    return grouped

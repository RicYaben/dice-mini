import pandas as pd

def count_groups(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    "returns a grouped dataframe with counts using a list of columns. The order of the columns determines how the groups are formed"
    
    # Group by the specified columns and count the occurrences
    grouped = df.groupby(list(cols)).size().to_frame('count')
    grouped = grouped.sort_values("count", ascending=False)  # Sort by index to keep order

    return grouped


def count_multi_groups(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Count rows grouped by `cols` and produce a hierarchical table with subtotals.
    - Subtotal rows ("-") are only added for groups with more than one member.
    - Totals appear first in their group.
    - Groups and leaves are sorted by count descending.
    """
    # Step 1: leaf counts
    leaf_counts = df.groupby(cols).size().reset_index(name='count')

    all_rows = []

    def add_subtotals(df_subset, level):
        col = cols[level]
        # compute total count per group
        group_sums = df_subset.groupby(col)['count'].sum().sort_values(ascending=False)
        for name in group_sums.index:
            group = df_subset[df_subset[col] == name]
            # subtotal if group has more than 1 member
            if len(group) > 1:
                subtotal = {c: "-" for c in cols}
                subtotal[col] = name
                for i in range(level):
                    subtotal[cols[i]] = group.iloc[0][cols[i]]
                subtotal['count'] = group['count'].sum()
                all_rows.append(pd.DataFrame([subtotal]))
            # process next level or leaf
            if level + 1 < len(cols):
                # sort subgroups by total count descending
                add_subtotals(group, level + 1)
            else:
                # leaf level: sort by count descending
                leaf_sorted = group.sort_values('count', ascending=False)
                all_rows.append(leaf_sorted)

    add_subtotals(leaf_counts, 0)

    combined = pd.concat(all_rows, ignore_index=True)
    combined['count'] = combined['count'].astype(int)

    return combined.set_index(cols)[['count']]

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

class Anonymizer:
    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cols:
            self._apply(df, col)
        return df

    def _apply(self, df: pd.DataFrame, col_path: str) -> None:
        parts = col_path.split(".")

        base_col = parts[0]
        nested_path = parts[1:]

        if not nested_path:
            df[base_col] = self._map_series(df[base_col])
            return

        # extract nested values
        def extract(x):
            for p in nested_path:
                if isinstance(x, dict):
                    x = x.get(p)
                else:
                    return None
            return x

        extracted = df[base_col].apply(extract)

        mapping = {
            v: i
            for i, v in enumerate(extracted.dropna().unique(), start=1)
        }

        # write back into original structure
        def update(row):
            obj = row[base_col]
            if not isinstance(obj, dict):
                return obj

            target = obj
            for p in parts[1:-1]:
                target = target.setdefault(p, {})

            leaf = parts[-1]
            val = obj.get(leaf)

            if val in mapping:
                obj[leaf] = mapping[val]

            return obj

        df[base_col] = df.apply(update, axis=1)

    def _map_series(self, series: pd.Series) -> pd.Series:
        mapping = {
            v: i
            for i, v in enumerate(series.dropna().unique(), start=1)
        }
        return series.map(mapping)

def new_anonymizer(cols: list[str]) -> Anonymizer:
    return Anonymizer(cols)

class FieldRemover:
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths

    def remove(self, df: pd.DataFrame) -> pd.DataFrame:
        for path in self.paths:
            self._remove(df, path)
        return df

    def _remove(self, df: pd.DataFrame, path: str) -> None:
        protocol, field_path = path.split(".", 1)
        field_parts = field_path.split(".")

        def remove_from_services(services):
            if not isinstance(services, list):
                return services

            for svc in services:
                if not isinstance(svc, dict):
                    continue

                if svc.get("protocol") != protocol:
                    continue

                data = svc.get("data")
                if not isinstance(data, dict):
                    continue

                self._delete_nested(data, field_parts)

            return services

        df["services"] = df["services"].apply(remove_from_services)

    def _delete_nested(self, obj: dict, parts: list[str]) -> None:
        current = obj

        for key in parts[:-1]:
            if not isinstance(current.get(key), dict):
                return
            current = current[key]

        current.pop(parts[-1], None)

def new_remover(cols: list[str]) -> FieldRemover:
    return FieldRemover(cols)
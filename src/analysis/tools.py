from pathlib import Path
from typing import Optional

import pandas as pd
import ujson

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
    def __init__(self, cols: list[str], out: Optional[str] = None) -> None:
        self.cols = cols
        self.mappings = {}
        self._out = None
        self.out = out

        if (o:=self.out) and o.exists():
            self.mappings = ujson.loads(o.read_text())
            return
        
    @property    
    def out(self):
        return self._out
    
    @out.setter
    def out(self, fpath: Optional[str]) -> None | Path:
        if fpath:
            self._out = Path(fpath)

    def _save(self):
        if self.out:
            self.out.write_text(ujson.dumps(self.mappings, indent=2))

    def anonymize(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cols:
            self._apply(df, col)
        self._save()
        return df
    
    def _update_nested(self, row, base_col, nested_path, field_key, extract_fn):
        obj = row[base_col]

        if not isinstance(obj, dict):
            return obj

        if field_key not in self.mappings:
            self.mappings[field_key] = {}

        mapping = self.mappings[field_key]
        next_id = max(mapping.values(), default=0) + 1

        val = extract_fn(obj)

        if val is None:
            return obj

        if val not in mapping:
            mapping[val] = next_id
            next_id += 1

        # write back into nested structure
        target = obj
        for p in nested_path[:-1]:
            target = target.setdefault(p, {})

        target[nested_path[-1]] = mapping[val]

        return obj

    def _map_series(self, series: pd.Series, field_key: str) -> pd.Series:
        if field_key not in self.mappings:
            self.mappings[field_key] = {}

        mapping = self.mappings[field_key]

        new_value = max(mapping.values(), default=0) + 1

        def map_value(v):
            if pd.isna(v):
                return v
            if v in mapping:
                return mapping[v]

            nonlocal new_value
            mapping[v] = new_value
            new_value += 1
            return mapping[v]

        return series.map(map_value)

    def _apply(self, df: pd.DataFrame, col_path: str) -> None:
        parts = col_path.split(".")
        base_col = parts[0]
        nested_path = parts[1:]
        field_key = col_path

        if not nested_path:
            df[base_col] = self._map_series(df[base_col], field_key)
            return

        leaf = nested_path[-1]

        def update(row):
            obj = row[base_col]

            if not isinstance(obj, dict):
                return obj

            if field_key not in self.mappings:
                self.mappings[field_key] = {}

            mapping = self.mappings[field_key]
            next_id = max(mapping.values(), default=0) + 1

            # walk to value
            val = obj
            for p in nested_path:
                if isinstance(val, dict):
                    val = val.get(p)
                else:
                    return obj

            if val is None:
                return obj

            if val not in mapping:
                mapping[val] = next_id

            # write back
            target = obj
            for p in nested_path[:-1]:
                target = target.setdefault(p, {})

            target[leaf] = mapping[val]

            return obj

        df[base_col] = df.apply(update, axis=1)

def new_anonymizer(cols: list[str], out: Optional[str]) -> Anonymizer:
    return Anonymizer(cols, out)

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
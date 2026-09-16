import pandas as pd


def pick_family_representatives(
    clusters: list[list[str]], importance: pd.Series
) -> list[str]:
    """One feature per cluster: the highest-importance member. Ties break
    on cluster order (first listed wins), so a rerun is reproducible rather
    than depending on pandas' internal tie-breaking."""
    reps = []
    for cluster in clusters:
        best = max(cluster, key=lambda c: (importance[c], -cluster.index(c)))
        reps.append(best)
    return reps


def cluster_correlated_features(
    df: pd.DataFrame, feature_cols: list[str], threshold: float = 0.8
) -> list[list[str]]:
    """Groups feature_cols into families by Spearman rank correlation.

    Two features join the same family if their absolute correlation is >=
    threshold. Uses union-find over pairwise comparisons, so a family can
    have more than 2 members via a chain (A~B, B~C -> {A,B,C}) even if A~C
    alone would fall under threshold.
    """
    corr = df[feature_cols].corr(method="spearman").abs()

    parent = {c: c for c in feature_cols}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, a in enumerate(feature_cols):
        for b in feature_cols[i + 1:]:
            if corr.loc[a, b] >= threshold:
                union(a, b)

    families: dict[str, list[str]] = {}
    for c in feature_cols:
        root = find(c)
        families.setdefault(root, []).append(c)

    return list(families.values())

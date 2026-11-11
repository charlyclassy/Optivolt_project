"""Portfolio summaries from original Optimise Prime data, no synthetic forecasts."""
from pathlib import Path
import pandas as pd

DEFAULT_CAPACITY = {
    "Bexleyheath":17.0, "Dartford":45.0, "Islington":13.0,
    "Mount Pleasant":260.0, "Orpington":14.0, "Premier Park":70.0,
    "Whitechapel":55.0, "Camden":21.0, "Victoria":29.0,
}

def available_depots(root: Path) -> list[str]:
    import depot_updates as dep
    return [name for name in DEFAULT_CAPACITY if name in dep.visible_depots()]

def portfolio_summary(root: Path) -> pd.DataFrame:
    rows = []
    visible = set(available_depots(root))
    for name, threshold in DEFAULT_CAPACITY.items():
        if name not in visible:
            continue
        path = root / f"UKPN_OptimisePrime_{name.replace(' ', '_')}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["timestamp", "demand_kw"], parse_dates=["timestamp"])
        vals = pd.to_numeric(df["demand_kw"], errors="coerce").dropna()
        if vals.empty:
            continue
        rows.append({
            "Depot":name,
            "Observed intervals":int(len(vals)),
            "Mean observed (kW)":round(float(vals.mean()),2),
            "Historical peak (kW)":round(float(vals.max()),2),
            "Planning threshold (kW)":threshold,
            "Historical peak / threshold (%)":round(float(vals.max()/threshold*100),1),
            "Last observed":str(df["timestamp"].max())[:16],
        })
    return pd.DataFrame(rows)

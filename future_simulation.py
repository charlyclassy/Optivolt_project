"""Deterministic scenario inputs for Optivolt future demand prediction.

These are simulated inputs derived from recorded historical depot readings;
never represent them as future measurements or independently validated forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_seed(observations: pd.DataFrame, start_date: str, growth_pct: float) -> pd.Series:
    """Build 96 quarter-hour inputs for the day before the forecast start.

    Uses the median of complete historical days with the same weekday as the
    seed date, prioritising the same calendar month for seasonal relevance.
    No randomness: identical observations and parameters give identical inputs.
    """
    if not np.isfinite(growth_pct) or growth_pct < -100:
        raise ValueError('Demand growth must be a finite percentage of at least -100%.')
    target = pd.Timestamp(start_date).normalize()
    if pd.isna(target):
        raise ValueError('A valid prediction start date is required.')
    seed_day = target - pd.Timedelta(days=1)
    df = observations[['timestamp', 'demand_kw']].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True).dt.tz_localize(None)
    df['demand_kw'] = pd.to_numeric(df['demand_kw'], errors='coerce')
    df = df.dropna().sort_values('timestamp').drop_duplicates('timestamp', keep='last')
    df = df[np.isfinite(df['demand_kw']) & (df['demand_kw'] >= 0)]
    df['day'] = df['timestamp'].dt.normalize()
    df['slot'] = df['timestamp'].dt.hour * 4 + df['timestamp'].dt.minute // 15
    df = df[(df['timestamp'].dt.minute % 15 == 0) & (df['timestamp'].dt.second == 0)]
    complete = df.groupby('day').filter(lambda x: len(x) == 96 and x['slot'].nunique() == 96)
    matched = complete[complete['day'].dt.weekday == seed_day.weekday()]
    seasonal = matched[matched['day'].dt.month == seed_day.month]
    if not seasonal.empty:
        matched = seasonal
    if matched.empty:
        raise ValueError('No complete historical 24-hour day matches the required weekday.')
    profile = matched.groupby('slot')['demand_kw'].median().reindex(range(96))
    if profile.isna().any():
        raise ValueError('Historical observations cannot form a complete 96-step profile.')
    values = (profile.to_numpy(dtype=np.float64) * (1 + growth_pct / 100)).clip(min=0)
    idx = pd.date_range(seed_day, periods=96, freq='15min')
    result = pd.Series(values, index=idx, name='synthetic_input_demand_kw')
    result.attrs['historical_reference_day'] = (
        f'median of {matched["day"].nunique()} complete historical days '
        f'(weekday matched; {"same month" if not seasonal.empty else "all available months"})'
    )
    return result

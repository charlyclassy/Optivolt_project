"""Inference for a separately trained, direct multi-output 30-day LSTM."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def model_available(depot):
    folder = ROOT / 'models' / 'long_term_lstm' / depot.replace(' ', '_').lower()
    return (folder / 'model.keras').is_file() and (folder / 'metadata.json').is_file()


def forecast(depot, history, hours):
    """Predict 15-minute values from the most recent 60 complete days.

    Forecasts are anchored to the last observation; future-date scenarios are
    intentionally unsupported because the trained model has no future covariates.
    """
    import tensorflow as tf
    if not 1 <= int(hours) <= 720:
        raise ValueError('Horizon must be between 1 and 720 hours.')
    folder = ROOT / 'models' / 'long_term_lstm' / depot.replace(' ', '_').lower()
    if not model_available(depot):
        raise FileNotFoundError(f'No trained 30-day LSTM for {depot}. Run train_long_term_lstm.py first.')
    meta = json.loads((folder / 'metadata.json').read_text())
    series = pd.Series(history).sort_index().resample('15min').mean().dropna()
    # Input must be aligned to a complete 60-day block ending at 23:45.
    if series.index[-1].hour != 23 or series.index[-1].minute != 45:
        raise ValueError('30-day LSTM requires the latest reading at 23:45 (complete day).')
    tail = series.tail(60 * 96)
    expected = pd.date_range(series.index[-1] - pd.Timedelta(minutes=15 * (60 * 96 - 1)),
                             periods=60 * 96, freq='15min')
    if len(tail) != len(expected) or not tail.index.equals(expected):
        raise ValueError('30-day LSTM requires 60 consecutive complete days of readings.')
    x = ((tail.to_numpy(dtype='float32').reshape(1, 60, 96) - meta['mean']) / meta['std'])
    model = tf.keras.models.load_model(folder / 'model.keras', compile=False)
    result = np.maximum(0, model.predict(x, verbose=0).reshape(-1) * meta['std'] + meta['mean'])
    n = int(hours) * 4
    dates = pd.date_range(series.index[-1] + pd.Timedelta(minutes=15), periods=n, freq='15min')
    return tail, pd.Series(result[:n], index=dates, name='Dedicated 30-day LSTM (kW)')

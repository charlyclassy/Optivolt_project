"""Evaluate SAVED 24h-input LSTM with genuinely recursive 72h forecasts.

Usage:
    python evaluate_short_term_lstm.py --depot Bexleyheath
    python evaluate_short_term_lstm.py --all

No training or model/scaler changes. Uses the final 15% of the same contiguous
complete-day data block selected by train_long_term_lstm.load_days. Evaluation
origins are 72h apart, so forecast targets do not overlap. Unlike the 30-day
script, test targets START after the 85% chronological cutoff. Different
forecast horizons/sampling schemes mean scores are descriptive, not a matched
head-to-head comparison.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from train_long_term_lstm import ROOT, load_days, STEPS_PER_DAY

DEPOTS = ('Bexleyheath', 'Camden', 'Dartford', 'Islington', 'Mount Pleasant',
          'Orpington', 'Premier Park', 'Victoria', 'Whitechapel')
HORIZON_DAYS = 3
HORIZON_STEPS = HORIZON_DAYS * STEPS_PER_DAY
LOOKBACK = STEPS_PER_DAY


def evaluate(depot):
    # Delayed import keeps --help fast and avoids loading TF before validation.
    import tensorflow as tf
    slug = depot.lower().replace(' ', '_').replace('-', '_')
    source = ROOT / f'UKPN_OptimisePrime_{depot.replace(" ", "_")}.csv'
    model_path = ROOT / 'models' / f'{slug}.keras'
    scaler_path = ROOT / 'models' / f'{slug}_scalers.npz'
    for path in (source, model_path, scaler_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}')

    days, dates = load_days(source)
    values = days.reshape(-1).astype(np.float64)
    test_start_day = int(len(days) * .85)
    origins = np.arange(test_start_day * STEPS_PER_DAY,
                        len(values) - HORIZON_STEPS + 1, HORIZON_STEPS,
                        dtype=int)
    if not len(origins):
        raise ValueError(f'{depot}: no complete held-out 72-hour windows')
    # Each origin is the first predicted timestamp, never an observed input.
    windows = np.stack([values[o-LOOKBACK:o] for o in origins]).astype(np.float32)
    actual = np.stack([values[o:o+HORIZON_STEPS] for o in origins])
    with np.load(scaler_path) as scalers:
        xm = float(np.asarray(scalers['x_mean']).reshape(-1)[0])
        xs = float(np.asarray(scalers['x_scale']).reshape(-1)[0])
        ym = float(np.asarray(scalers['y_mean']).reshape(-1)[0])
        ys = float(np.asarray(scalers['y_scale']).reshape(-1)[0])
    if xs <= 0 or ys <= 0:
        raise ValueError('Invalid saved StandardScaler scale')
    model = tf.keras.models.load_model(model_path, compile=False)
    predicted = np.empty_like(actual)
    # Batch the independent forecast origins; feed every predicted demand
    # back into its own input window for all 288 forecast steps.
    @tf.function(input_signature=[tf.TensorSpec((None, LOOKBACK, 1), tf.float32)])
    def step(batch):
        return model(batch, training=False)
    for t in range(HORIZON_STEPS):
        inputs = ((windows - xm) / xs).astype(np.float32)[..., None]
        scaled = step(tf.convert_to_tensor(inputs)).numpy().reshape(-1)
        next_values = np.maximum(0., scaled.astype(np.float64) * ys + ym)
        predicted[:, t] = next_values
        windows[:, :-1] = windows[:, 1:]
        windows[:, -1] = next_values

    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1. - np.sum(residual ** 2) / denominator) if denominator > 0 else None
    mask = np.abs(actual) > .1
    mape = float(100. * np.mean(np.abs(residual[mask] / actual[mask]))) if mask.any() else None
    peak_error = float(np.mean(np.abs(predicted.max(axis=1) - actual.max(axis=1))))
    result = {
        'depot': depot,
        'model_type': 'Existing saved 96-step LSTM, recursive 72-hour forecast',
        'input_steps': LOOKBACK,
        'output_steps': HORIZON_STEPS,
        'test_windows': int(len(origins)),
        'test_points': int(actual.size),
        'test_start': str(dates[test_start_day]),
        'test_last_target': str(dates[-1]),
        'test_mae_kw': mae,
        'test_rmse_kw': rmse,
        'test_r2': r2,
        'test_mape_pct': mape,
        'test_peak_error_kw': peak_error,
        'mape_min_actual_kw': .1,
        'peak_error_definition': 'Mean absolute peak error across non-overlapping held-out 72-hour windows',
        'evaluation_scope': 'Last 15% of contiguous complete-day history; non-overlapping 72-hour recursive forecasts; targets entirely in test period',
        'comparison_note': 'Different horizon and test-window sampling from 30-day LSTM; scores are not a matched head-to-head test.',
    }
    out = ROOT / 'models' / 'short_term_lstm_evaluation' / f'{slug}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'\n{depot}: saved {out.relative_to(ROOT)}')
    print(json.dumps({k: result[k] for k in ('test_windows', 'test_mae_kw', 'test_rmse_kw',
          'test_r2', 'test_mape_pct', 'test_peak_error_kw')}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--depot', help='Depot name, e.g. "Mount Pleasant"')
    selection.add_argument('--all', action='store_true', help='Evaluate all nine depots')
    args = parser.parse_args()
    for depot in (DEPOTS if args.all else (args.depot,)):
        evaluate(depot)


if __name__ == '__main__':
    main()

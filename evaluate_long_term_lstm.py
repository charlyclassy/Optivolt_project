"""Evaluate a SAVED 30-day LSTM on the same chronological holdout as training.
No retraining or overwriting of model weights occurs.
Usage: python evaluate_long_term_lstm.py --depot Bexleyheath
"""
import argparse
import json
from pathlib import Path
import numpy as np
from train_long_term_lstm import ROOT, INPUT_DAYS, OUTPUT_DAYS, load_days, windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depot', required=True)
    args = parser.parse_args()
    import keras
    slug = args.depot.replace(' ', '_')
    folder = ROOT / 'models' / 'long_term_lstm' / slug.lower()
    meta_path = folder / 'metadata.json'
    meta = json.loads(meta_path.read_text())
    days, _ = load_days(ROOT / f'UKPN_OptimisePrime_{slug}.csv')
    x, y, ends = windows(days)
    val_cut = int(len(days) * .85)
    test = ends >= val_cut
    if not test.any():
        raise ValueError('No held-out test windows')
    mean, std = float(meta['mean']), float(meta['std'])
    x = (x[test] - mean) / std
    actual = y[test]
    model = keras.models.load_model(folder / 'model.keras', compile=False)
    predicted = np.maximum(0, model.predict(x, verbose=0) * std + mean)
    residual = predicted - actual
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - np.sum(residual ** 2) / denominator) if denominator > 0 else None
    mask = np.abs(actual) > .1
    mape = float(np.mean(np.abs(residual[mask] / actual[mask])) * 100) if mask.any() else None
    peak_error = float(np.mean(np.abs(predicted.max(axis=1) - actual.max(axis=1))))
    meta.update(test_mae_kw=mae, test_rmse_kw=rmse, test_r2=r2,
                test_mape_pct=mape, test_peak_error_kw=peak_error,
                mape_min_actual_kw=.1,
                peak_error_definition='Mean absolute peak error across held-out 30-day windows')
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ('test_mae_kw', 'test_rmse_kw',
          'test_r2', 'test_mape_pct', 'test_peak_error_kw')}, indent=2))

if __name__ == '__main__':
    main()

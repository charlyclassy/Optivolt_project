"""Train experimental Optivolt LSTMs with real UKPN + 365 synthetic days.

Never overwrites original models. Synthetic data contributes to training only.
Validation/test use real observed UKPN readings. Separate real/synthetic windows
avoid crossing the artificial 2030 timestamp boundary.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEPOTS = ('Bexleyheath', 'Camden', 'Dartford', 'Islington', 'Mount Pleasant', 'Orpington')
STEPS = 96


def read_real(path):
    """Keep the contiguous time grid; track which targets are genuinely observed."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df['demand_kw'] = pd.to_numeric(df['demand_kw'], errors='coerce')
    df = df.dropna(subset=['timestamp']).drop_duplicates('timestamp', keep='last').set_index('timestamp').sort_index()
    series = df['demand_kw'].resample('15min').mean()
    if 'observed' in df:
        observed = df['observed'].astype(str).str.strip().str.lower().isin(('1', 'true', 'yes'))
        observed = observed.resample('15min').min().reindex(series.index, fill_value=False).astype(bool)
    else:
        observed = pd.Series(True, index=series.index)
    valid = series.notna() & np.isfinite(series) & series.ge(0)
    complete = valid.resample('D').sum().eq(STEPS)
    dates = complete[complete].index
    if len(dates) == 0:
        raise ValueError('No complete days with valid demand readings')
    groups = dates.to_series().diff().dt.days.ne(1).cumsum()
    blocks = [g.index for _, g in dates.to_series().groupby(groups)]
    block = max(blocks, key=len)
    if len(block) < 180:
        raise ValueError(f'Longest continuous valid-demand block is {len(block)} days; need at least 180')
    idx = pd.date_range(block[0], periods=len(block)*STEPS, freq='15min')
    values = series.reindex(idx).to_numpy(dtype=np.float32).reshape(-1, STEPS)
    flags = observed.reindex(idx, fill_value=False).to_numpy(dtype=bool).reshape(-1, STEPS)
    return values, block, flags


def read_synthetic(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    if len(df) != 365 * STEPS:
        raise ValueError(f'Expected 35,040 synthetic rows, found {len(df)}')
    df = df.sort_values('timestamp')
    if df['timestamp'].duplicated().any() or not (df['timestamp'].diff().iloc[1:] == pd.Timedelta(minutes=15)).all():
        raise ValueError('Synthetic timestamps are not a contiguous 15-minute series')
    values = pd.to_numeric(df['demand_kw'], errors='coerce').to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid synthetic demand')
    return values.reshape(365, STEPS)


def long_windows(days):
    x, y, ends = [], [], []
    for i in range(len(days) - 60 - 30 + 1):
        x.append(days[i:i + 60]); y.append(days[i + 60:i + 90].reshape(-1)); ends.append(i + 89)
    return np.asarray(x), np.asarray(y), np.asarray(ends)


def short_windows(days):
    # Each input contains the prior 96 points; target is the next 15-minute reading.
    a = days.reshape(-1)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(a, 97)
    return w[:, :96].copy(), w[:, 96].copy(), np.arange(96, len(a))


def metrics(actual, pred, peaks=None):
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    error = pred - actual
    den = np.sum((actual - actual.mean()) ** 2)
    mask = actual > 0.1
    out = {'test_mae_kw': float(np.abs(error).mean()),
           'test_rmse_kw': float(np.sqrt((error ** 2).mean())),
           'test_r2': float(1 - (error ** 2).sum() / den) if den > 0 else None,
           'test_mape_pct': float(100 * np.mean(np.abs(error[mask] / actual[mask]))) if mask.any() else None}
    if peaks is not None:
        out['test_peak_error_kw'] = float(np.mean(np.abs(pred.max(axis=1) - actual.max(axis=1))))
    return out


def train_one(depot, kind, epochs, batch_size):
    import keras
    import tensorflow as tf
    slug = depot.replace(' ', '_')
    real_path = ROOT / f'UKPN_OptimisePrime_{slug}.csv'
    synthetic_path = ROOT / 'synthetic_training_data' / f'UKPN_OptimisePrime_{slug}_synthetic_365d.csv'
    if not real_path.is_file() or not synthetic_path.is_file():
        raise FileNotFoundError(f'Missing real or synthetic CSV for {depot}')
    real, dates, observed = read_real(real_path)
    synth = read_synthetic(synthetic_path)
    train_cut, val_cut = int(len(real) * .70), int(len(real) * .85)
    # This matches the generator's 70%-of-longest-real-block source cutoff.
    if train_cut < 14:
        raise ValueError('Too few observed training days')
    mean = float(real[:train_cut].mean())
    std = max(float(real[:train_cut].std()), 1e-6)
    output = ROOT / 'models' / 'augmented_experiments' / ('long_term_lstm' if kind == 'long' else 'short_term_lstm') / slug.lower()
    if output.exists():
        raise FileExistsError(f'{output} already exists; move or rename it before retraining.')
    keras.utils.set_random_seed(42)
    if kind == 'long':
        rx, ry, ends = long_windows(real)
        _, observed_targets, _ = long_windows(observed)
        sx, sy, _ = long_windows(synth)
        train = ends < train_cut
        val = (ends >= train_cut) & (ends < val_cut)
        test = ends >= val_cut
        if min(train.sum(), val.sum(), test.sum()) < 10:
            raise ValueError('Insufficient real 30-day windows for split')
        xtrain = np.concatenate([rx[train], sx]); ytrain = np.concatenate([ry[train], sy])
        xval, yval = rx[val], ry[val]
        xtest, ytest = rx[test], ry[test]
        # Zero-weight unobserved real targets in training and validation.
        real_train_weights = observed_targets[train].astype(np.float32)
        train_weights = np.concatenate([real_train_weights, np.ones_like(sy, dtype=np.float32)])
        val_weights = observed_targets[val].astype(np.float32)
        test_mask = observed_targets[test].astype(bool)
        if not test_mask.any() or not val_weights.any():
            raise ValueError("No observed real validation/test targets for long-term evaluation")
        model = keras.Sequential([keras.layers.Input((60, 96)), keras.layers.LSTM(48),
                                  keras.layers.Dense(256, activation='relu'),
                                  keras.layers.Dropout(.20), keras.layers.Dense(30 * 96)])
        xtrain = (xtrain - mean) / std; ytrain = (ytrain - mean) / std
        xval = (xval - mean) / std; yval = (yval - mean) / std
        xtest = (xtest - mean) / std
    else:
        rx, ry, ends = short_windows(real)
        _, observed_targets, _ = short_windows(observed)
        sx, sy, _ = short_windows(synth)
        train = (ends < train_cut * STEPS) & observed_targets
        val = (ends >= train_cut * STEPS) & (ends < val_cut * STEPS) & observed_targets
        test = (ends >= val_cut * STEPS) & observed_targets
        xtrain = np.concatenate([rx[train], sx]); ytrain = np.concatenate([ry[train], sy])
        xval, yval = rx[val], ry[val]
        xtest, ytest = rx[test], ry[test]
        model = keras.Sequential([keras.layers.Input((96, 1)),
                                  keras.layers.LSTM(64, return_sequences=True),
                                  keras.layers.LSTM(32), keras.layers.Dense(1)])
        xtrain = ((xtrain - mean) / std)[..., None]; ytrain = (ytrain - mean) / std
        xval = ((xval - mean) / std)[..., None]; yval = (yval - mean) / std
        xtest = ((xtest - mean) / std)[..., None]
    if min(int(train.sum()), int(val.sum()), int(test.sum())) < 10:
        raise ValueError("Insufficient real observed training, validation or test targets")
    model.compile(optimizer=keras.optimizers.Adam(.001), loss='mse')
    print(f'\n{depot} {kind}: real train={int(train.sum())}, synthetic train={len(sx)}, real val={int(val.sum())}, real test={int(test.sum())}', flush=True)
    model.fit(xtrain.astype('float32'), ytrain.astype('float32'),
              validation_data=(xval.astype('float32'), yval.astype('float32')),
              epochs=epochs, batch_size=batch_size, shuffle=False, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)])
    pred = np.maximum(0, model.predict(xtest.astype('float32'), verbose=0) * std + mean)
    result = metrics(ytest[test_mask], pred[test_mask]) if kind == "long" else metrics(ytest, pred)
    if kind == "long":
        per_window = test_mask.any(axis=1)
        result["test_peak_error_kw"] = float(np.mean([abs(pred[i, test_mask[i]].max() - ytest[i, test_mask[i]].max()) for i in np.flatnonzero(per_window)]))
        result["observed_test_readings"] = int(test_mask.sum())
    if kind == 'long':
        baseline = np.maximum(0, rx[test, -30:].reshape(-1, 2880))
        result['test_baseline_mae_kw'] = float(np.mean(np.abs(baseline[test_mask] - ytest[test_mask])))
    else:
        # Previous-day baseline on real test targets.
        full = real.reshape(-1)
        targets = ends[test]
        baseline = full[targets - 96]
        result['test_baseline_mae_kw'] = float(np.mean(np.abs(baseline - ytest)))
        # 72-hour recursive forecast from nonoverlapping real held-out origins.
        origins = np.arange(val_cut * STEPS, len(full) - 288 + 1, 288, dtype=int)
        if len(origins):
            history = np.stack([full[o - 96:o] for o in origins]).astype(np.float32)
            actual72 = np.stack([full[o:o + 288] for o in origins])
            predictions72 = np.empty_like(actual72)
            @tf.function(input_signature=[tf.TensorSpec((None, 96, 1), tf.float32)])
            def step(x):
                return model(x, training=False)
            for t in range(288):
                out = step(tf.convert_to_tensor(((history - mean) / std)[..., None], tf.float32)).numpy().reshape(-1)
                next_y = np.maximum(0, out * std + mean)
                predictions72[:, t] = next_y
                history[:, :-1] = history[:, 1:]
                history[:, -1] = next_y
            actual_flags = np.stack([observed.reshape(-1)[o:o + 288] for o in origins])
            if actual_flags.any():
                rec = metrics(actual72[actual_flags], predictions72[actual_flags])
                valid_windows = actual_flags.any(axis=1)
                rec['test_peak_error_kw'] = float(np.mean([abs(actual72[i, actual_flags[i]].max() - predictions72[i, actual_flags[i]].max()) for i in np.flatnonzero(valid_windows)]))
                result['recursive_72h'] = {'test_windows': int(valid_windows.sum()), 'observed_test_readings': int(actual_flags.sum()), **rec}
    output = ROOT / 'models' / 'augmented_experiments' / ('long_term_lstm' if kind == 'long' else 'short_term_lstm') / slug.lower()
    # Fail rather than silently replacing a prior experimental run.
    if output.exists():
        raise FileExistsError(f'{output} exists. Move or rename it before retraining; nothing overwritten.')
    output.mkdir(parents=True)
    model.save(output / 'model.keras')
    if kind == 'short':
        np.savez_compressed(output / 'scalers.npz', x_mean=[mean], x_scale=[std], y_mean=[mean], y_scale=[std])
    result.update({'depot': depot, 'model_kind': kind, 'source_real': real_path.name,
                   'source_synthetic': str(synthetic_path.relative_to(ROOT)),
                   'synthetic_days': 365, 'real_days': len(real),
                   'train_real_windows': int(train.sum()), 'train_synthetic_windows': len(sx),
                   'validation_real_windows': int(val.sum()), 'test_real_windows': int(test.sum()),
                   'real_train_cutoff': str(dates[train_cut - 1]),
                   'real_test_start': str(dates[val_cut]), 'real_last_day': str(dates[-1]),
                   'mean': mean, 'std': std,
                   'note': 'Synthetic data for fitting only. Real demand with observed=0 can appear as input context, but only observed=1 real targets contribute to training loss, validation and reported test metrics. Synthetic generator provenance must be independently checked against real training cutoff. Model not activated.'})
    (output / 'metadata.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print(f'SAVED: {output}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--depot', choices=DEPOTS, required=True)
    p.add_argument('--model', choices=('short', 'long', 'both'), default='both')
    p.add_argument('--epochs', type=int, default=35)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    for kind in (('short', 'long') if args.model == 'both' else (args.model,)):
        train_one(args.depot, kind, args.epochs, args.batch_size)


if __name__ == '__main__':
    main()

"""Train a separate 60-day-input / 30-day-output LSTM for one depot.

Usage: python train_long_term_lstm.py --depot Bexleyheath --epochs 35
Requires TensorFlow and the repository's UKPN_OptimisePrime_<Depot>.csv files.
This script DOES NOT overwrite the existing 72-hour LSTM.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STEPS_PER_DAY = 96
INPUT_DAYS = 60
OUTPUT_DAYS = 30


def load_days(csv_path):
    frame = pd.read_csv(csv_path, parse_dates=['timestamp'])
    frame['demand_kw'] = pd.to_numeric(frame['demand_kw'], errors='coerce')
    series = (frame.dropna(subset=['timestamp']).drop_duplicates('timestamp', keep='last')
              .set_index('timestamp')['demand_kw'].sort_index())
    series = series.resample('15min').mean()
    # Preserve only complete, observed days; never silently fill a large gap.
    day_counts = series.resample('D').count()
    complete_days = day_counts[day_counts == STEPS_PER_DAY].index
    if len(complete_days) < INPUT_DAYS + OUTPUT_DAYS + 90:
        raise ValueError('Not enough complete days for a meaningful temporal holdout.')
    # Restrict to the most recent uninterrupted block of complete days.
    groups = (complete_days.to_series().diff().dt.days.ne(1)).cumsum()
    blocks = [g.index for _, g in complete_days.to_series().groupby(groups)]
    block = max(blocks, key=len)
    if len(block) < INPUT_DAYS + OUTPUT_DAYS + 90:
        raise ValueError('No sufficiently long uninterrupted run of complete days.')
    values = series.loc[block[0]:block[-1] + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)]
    return values.to_numpy(dtype='float32').reshape(-1, STEPS_PER_DAY), block


def windows(days):
    xs, ys, ends = [], [], []
    for start in range(len(days) - INPUT_DAYS - OUTPUT_DAYS + 1):
        xs.append(days[start:start + INPUT_DAYS])
        ys.append(days[start + INPUT_DAYS:start + INPUT_DAYS + OUTPUT_DAYS].reshape(-1))
        ends.append(start + INPUT_DAYS + OUTPUT_DAYS - 1)
    return np.asarray(xs), np.asarray(ys), np.asarray(ends)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depot', required=True, help='e.g. Bexleyheath or Mount Pleasant')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    import keras
    keras.utils.set_random_seed(42)
    slug = args.depot.replace(' ', '_')
    source = ROOT / f'UKPN_OptimisePrime_{slug}.csv'
    if not source.exists():
        raise FileNotFoundError(source)
    days, dates = load_days(source)
    x, y, ends = windows(days)
    # Chronological split by TARGET end date, not random shuffling.
    train_cut = int(len(days) * .70)
    val_cut = int(len(days) * .85)
    train = ends < train_cut
    val = (ends >= train_cut) & (ends < val_cut)
    test = ends >= val_cut
    if min(train.sum(), val.sum(), test.sum()) < 10:
        raise ValueError('Insufficient training/validation/test windows.')
    mean = float(days[:train_cut].mean())
    std = max(float(days[:train_cut].std()), 1e-6)
    x = (x - mean) / std
    y = (y - mean) / std
    model = keras.Sequential([
        keras.layers.Input(shape=(INPUT_DAYS, STEPS_PER_DAY)),
        keras.layers.LSTM(48),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(.20),
        keras.layers.Dense(OUTPUT_DAYS * STEPS_PER_DAY),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001), loss='mse', metrics=['mae'])
    model.fit(x[train], y[train], validation_data=(x[val], y[val]), epochs=args.epochs,
              batch_size=args.batch_size, shuffle=False, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=7,
                                                           restore_best_weights=True)])
    prediction = np.maximum(0, model.predict(x[test], verbose=0) * std + mean)
    actual = y[test] * std + mean
    mae = float(np.mean(np.abs(prediction - actual)))
    rmse = float(np.sqrt(np.mean((prediction - actual)**2)))
    # Persistence benchmark: repeat the last 30 observed days in the 60-day input.
    persistence = np.maximum(0, (x[test, -OUTPUT_DAYS:] * std + mean).reshape(-1, OUTPUT_DAYS * STEPS_PER_DAY))
    baseline_mae = float(np.mean(np.abs(persistence - actual)))
    output = ROOT / 'models' / 'long_term_lstm' / slug.lower()
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / 'model.keras')
    metadata = {'depot': args.depot, 'input_days': INPUT_DAYS, 'output_days': OUTPUT_DAYS,
                'steps_per_day': STEPS_PER_DAY, 'mean': mean, 'std': std,
                'test_mae_kw': mae, 'test_rmse_kw': rmse,
                'test_baseline_mae_kw': baseline_mae,
                'test_windows': int(test.sum()), 'train_windows': int(train.sum()),
                'validation_windows': int(val.sum()), 'last_observation': str(dates[-1])}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    if mae >= baseline_mae:
        print('WARNING: Model did not beat the persistence baseline on the held-out test windows.')

if __name__ == '__main__':
    main()

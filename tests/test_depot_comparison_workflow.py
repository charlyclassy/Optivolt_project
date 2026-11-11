"""Fast integration checks for depot-scoped comparison persistence without TensorFlow runtime."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class FakeRegressor:
    def fit(self, x, y, **kwargs):
        self.mean = float(np.mean(y))
        return self

    def predict(self, x, **kwargs):
        return np.full(len(x), self.mean)


class FakeHistory:
    history = {'loss': [0.8, 0.5], 'val_loss': [0.9, 0.6]}


class FakeNetwork(FakeRegressor):
    def compile(self, **kwargs):
        pass

    def fit(self, x, y, **kwargs):
        super().fit(x, y)
        return FakeHistory()

    def predict(self, x, **kwargs):
        return super().predict(x).reshape(-1, 1)


def load_module(monkeypatch, tmp_path):
    keras = types.SimpleNamespace(
        utils=types.SimpleNamespace(set_random_seed=lambda seed: None),
        layers=types.SimpleNamespace(Input=lambda *a: None, LSTM=lambda *a: None, Dense=lambda *a: None),
        Sequential=lambda layers: FakeNetwork(),
        callbacks=types.SimpleNamespace(EarlyStopping=lambda **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, 'tensorflow', types.SimpleNamespace(keras=keras))
    source = Path(__file__).resolve().parents[1] / 'depot_model_comparison.py'
    spec = importlib.util.spec_from_file_location('depot_model_comparison_under_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(module, 'RandomForestRegressor', lambda **kwargs: FakeRegressor())
    monkeypatch.setattr(module, 'XGBRegressor', lambda **kwargs: FakeRegressor())
    return module


def readings(start, n=1200, shift=0):
    idx = pd.date_range(start, periods=n, freq='15min')
    return pd.Series(5 + shift + np.sin(np.arange(n) / 10), index=idx)


def test_comparisons_are_depot_scoped_versioned_and_fresh(monkeypatch, tmp_path):
    m = load_module(monkeypatch, tmp_path)
    a, b = readings('2026-01-01'), readings('2026-02-01', shift=3)
    result_a = m.run('New8', a, 10, min_targets=1000, epochs=2)
    result_b = m.run('Bexleyheath', b, 12, min_targets=1000, epochs=2)
    assert m.load('New8')['metadata']['depot'] == 'New8'
    assert m.load('Bexleyheath')['metadata']['depot'] == 'Bexleyheath'
    assert result_a['metadata']['data_fingerprint'] != result_b['metadata']['data_fingerprint']
    assert len(result_a['metrics']) == 4
    assert len(result_a['predictions']) == result_a['metadata']['test_targets']
    assert (tmp_path / 'new8' / 'versions' / result_a['metadata']['run_id'] / 'metadata.json').exists()
    assert m.data_fingerprint(a) != m.data_fingerprint(a + 1)
    assert m.data_fingerprint(a) != m.data_fingerprint(a.iloc[:-1])
    assert m.data_fingerprint(a) != m.data_fingerprint(a.set_axis(a.index + pd.Timedelta(minutes=15)))


def test_insufficient_and_gapped_data_rejected(monkeypatch, tmp_path):
    m = load_module(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match='eligible consecutive-window targets'):
        m.run('New8', readings('2026-01-01', n=500), 10, min_targets=1000)
    with pytest.raises(ValueError, match='between 1,000 and 200,000'):
        m.run('New8', readings('2026-01-01'), 10, min_targets=999)
    gapped = readings('2026-01-01').drop(readings('2026-01-01').index[400:650])
    with pytest.raises(ValueError, match='eligible consecutive-window targets'):
        m.run('New8', gapped, 10, min_targets=1000)

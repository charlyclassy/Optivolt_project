"""Isolated checks for borrowed-model artifact freshness without loading TensorFlow."""
import ast
import hashlib
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / 'optivolt_lstm.py'


def _signature_function(tmp_path):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_saved_lstm_artifact_signature')
    scope = {'MODEL_DIR': tmp_path, '_model_slug': lambda depot: depot.lower(),
             'hashlib': hashlib}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['_saved_lstm_artifact_signature']


def test_missing_artifacts_do_not_crash(tmp_path):
    signature = _signature_function(tmp_path)
    assert signature('New8') is None
    (tmp_path / 'new8.keras').write_bytes(b'weights')
    assert signature('New8') is None


def test_replacing_weights_or_scalers_changes_signature(tmp_path):
    signature = _signature_function(tmp_path)
    weights = tmp_path / 'new8.keras'
    scalers = tmp_path / 'new8_scalers.npz'
    weights.write_bytes(b'weights1')
    scalers.write_bytes(b'scalers1')
    initial = signature('New8')
    assert initial is not None
    weights.write_bytes(b'weights-replacement')
    after_weights = signature('New8')
    assert after_weights != initial
    scalers.write_bytes(b'scalers-replacement')
    assert signature('New8') != after_weights
    scalers.unlink()
    assert signature('New8') is None

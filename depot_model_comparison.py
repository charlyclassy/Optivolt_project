"""Explicit, depot-specific four-model experiment. Never trains on page load."""
from pathlib import Path
import json, re, time, hashlib, shutil
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from tensorflow import keras

OUTPUT_DIR = Path(__file__).resolve().parent / 'data' / 'depot_comparisons'
LOOKBACK = 96

def _slug(name):
    return re.sub(r'[^a-z0-9_-]+', '_', str(name).lower()).strip('_')

def _paths(depot):
    p = OUTPUT_DIR / _slug(depot)
    return p, p / 'metrics.csv', p / 'predictions.csv', p / 'training_history.csv', p / 'metadata.json'

def data_fingerprint(history):
    """Fingerprint the complete selected-depot demand series, including timestamps."""
    series = pd.Series(history, dtype=float).sort_index()
    series.index = pd.to_datetime(series.index)
    series = series[~series.index.duplicated(keep='last')]
    hashed = pd.util.hash_pandas_object(series, index=True).to_numpy(dtype='uint64')
    return hashlib.sha256(hashed.tobytes()).hexdigest()

def load(depot):
    p, metrics, preds, hist, meta = _paths(depot)
    if not (metrics.exists() and preds.exists() and meta.exists()):
        return None
    metadata = json.loads(meta.read_text(encoding='utf-8'))
    if metadata.get('depot') != str(depot):
        raise ValueError('Saved comparison metadata belongs to a different depot.')
    return {'metrics':pd.read_csv(metrics), 'predictions':pd.read_csv(preds),
            'history':pd.read_csv(hist) if hist.exists() else pd.DataFrame(), 'metadata':metadata}

def list_versions(depot):
    """Return only this depot's completed immutable experiment versions."""
    folder = _paths(depot)[0] / 'versions'
    versions = []
    for meta_file in sorted(folder.glob('*/metadata.json'), reverse=True):
        try:
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
            if meta.get('depot') == str(depot) and all((meta_file.parent / name).exists()
                for name in ('metrics.csv', 'predictions.csv')):
                versions.append(meta)
        except (OSError, ValueError):
            continue
    return versions


def load_version(depot, run_id):
    """Read a historical version without altering the latest saved comparison."""
    folder = _paths(depot)[0] / 'versions' / str(run_id)
    meta_file = folder / 'metadata.json'
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text(encoding='utf-8'))
    if meta.get('depot') != str(depot) or meta.get('run_id') != str(run_id):
        raise ValueError('Experiment version does not belong to the selected depot.')
    return {'metadata':meta, 'metrics':pd.read_csv(folder / 'metrics.csv'),
            'predictions':pd.read_csv(folder / 'predictions.csv'),
            'history':pd.read_csv(folder / 'training_history.csv') if (folder / 'training_history.csv').exists() else pd.DataFrame()}


def comparison_commentary(metrics):
    """Explain measured winners, with no preference or fabricated score for LSTM."""
    directions = {'MAE (kW)':False, 'RMSE (kW)':False, 'R²':True,
                  'Breach recall (%)':True, 'False alarm ratio (%)':False,
                  'Breach precision (%)':True, 'F1-score (%)':True,
                  'Balanced accuracy (%)':True, 'Specificity (%)':True,
                  'Mean bias error (kW)':None}
    findings = []
    for column, higher in directions.items():
        if column not in metrics.columns:
            continue
        vals = pd.to_numeric(metrics[column], errors='coerce')
        if higher is None:
            vals = vals.abs()
        valid = vals.dropna()
        if valid.empty:
            findings.append(f'{column}: cannot determine a winner from this evaluation.')
            continue
        best = valid.max() if higher else valid.min()
        winners = metrics.loc[valid.index[valid == best], 'Model'].astype(str).tolist()
        findings.append(f"{column}: {' and '.join(winners)} {'tie for best' if len(winners)>1 else 'performs best'} ({float(best):.3f}{' absolute bias' if higher is None else ''}).")
    if 'LSTM' in metrics['Model'].values and 'Previous-day baseline' in metrics['Model'].values:
        lstm = metrics.loc[metrics['Model']=='LSTM'].iloc[0]
        baseline = metrics.loc[metrics['Model']=='Previous-day baseline'].iloc[0]
        for col in ('MAE (kW)', 'RMSE (kW)'):
            a, b = pd.to_numeric(lstm[col],errors='coerce'), pd.to_numeric(baseline[col],errors='coerce')
            if np.isfinite(a) and np.isfinite(b) and b > 0:
                delta = 100*(b-a)/b
                findings.append(f'LSTM {col} versus previous-day baseline: {abs(delta):.1f}% {"lower" if delta>=0 else "higher"}.')
    return findings


def synthetic_history(history, count=3000):
    """Clearly labelled synthetic 15-minute demand, never represented as measured data."""
    observed = pd.Series(history, dtype=float).dropna()
    if observed.empty:
        raise ValueError("A selected depot demand series is needed to set synthetic scale.")
    rng = np.random.default_rng(42)
    n = max(1200, int(count))
    step = np.arange(n)
    baseline = max(0.1, float(observed.median()))
    daily = np.sin(2*np.pi*(step % 96)/96 - 1.2)
    weekly = np.sin(2*np.pi*step/(96*7))
    values = np.maximum(0, baseline*(1 + .28*daily + .12*weekly) + rng.normal(0, max(.02,baseline*.035), n))
    start = pd.Timestamp('2026-01-01')
    return pd.Series(values, index=pd.date_range(start, periods=n, freq='15min'), name='Demand_kW')


def inspect(history, min_targets=1000):
    series = pd.Series(history, dtype=float).sort_index()
    series.index = pd.to_datetime(series.index)
    series = series[~series.index.duplicated(keep='last')]
    finite = np.isfinite(series.to_numpy())
    contiguous = np.asarray(series.index[1:] - series.index[:-1] == pd.Timedelta(minutes=15))
    valid = np.zeros(len(series), dtype=bool)
    streak = 0
    for i in range(len(series)):
        streak = streak+1 if finite[i] and (i == 0 or contiguous[i-1]) else (1 if finite[i] else 0)
        valid[i] = streak >= LOOKBACK + 1
    count = int(valid.sum())
    return {'rows':len(series), 'valid_rows':int(finite.sum()), 'gaps':int((~contiguous).sum()),
            'eligible_targets':count, 'required':int(min_targets), 'ready':count>=int(min_targets)}


def active_selection(depot):
    path = _paths(depot)[0] / 'active_model.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def activate(depot, model_name, run_id):
    allowed = ('Previous-day baseline', 'Random Forest', 'XGBoost', 'LSTM')
    if model_name not in allowed:
        raise ValueError('Unknown model.')
    folder = _paths(depot)[0] / 'versions' / str(run_id)
    meta_path = folder / 'metadata.json'
    if not meta_path.exists():
        raise ValueError('Saved model version is missing.')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta['depot'] != str(depot) or meta['run_id'] != str(run_id):
        raise ValueError('Selected model does not belong to this depot.')
    artifact = {'Random Forest':'random_forest.joblib','XGBoost':'xgboost.json','LSTM':'lstm.keras'}.get(model_name)
    if meta.get('source') == 'synthetic':
        raise ValueError('Synthetic-data models cannot be activated for operational forecasting.')
    if artifact and not (folder / artifact).exists():
        raise ValueError('This model has no saved weights; retrain it before activation.')
    selection = {'depot':str(depot),'model':model_name,'run_id':str(run_id),'source':meta.get('source','historical')}
    target = _paths(depot)[0] / 'active_model.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    temp.replace(target)
    return selection


def forecast_selected(depot, history, steps):
    """Recursive 15-minute forecast from an explicitly activated comparison model."""
    selection = active_selection(depot)
    if not selection:
        raise ValueError('No comparison model has been activated.')
    folder = _paths(depot)[0] / 'versions' / selection['run_id']
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('source') == 'synthetic':
        raise ValueError('Synthetic-only models cannot be used for operational forecasts.')
    name = selection['model']
    if name == 'Random Forest': model = joblib.load(folder/'random_forest.joblib')
    elif name == 'XGBoost':
        model = XGBRegressor(); model.load_model(str(folder/'xgboost.json'))
    elif name == 'LSTM':
        model = keras.models.load_model(folder/'lstm.keras')
        xsc = joblib.load(folder/'x_scaler.joblib'); ysc = joblib.load(folder/'y_scaler.joblib')
    values = pd.Series(history, dtype=float).sort_index().to_numpy()[-LOOKBACK:].tolist()
    if len(values) < LOOKBACK or not np.isfinite(values).all():
        raise ValueError('At least 96 valid recent readings are needed.')
    out=[]
    for _ in range(int(steps)):
        if name == 'Previous-day baseline': prediction = values[-LOOKBACK]
        elif name == 'LSTM':
            scaled = xsc.transform(np.asarray(values[-LOOKBACK:],dtype=float).reshape(1,-1))
            prediction = float(ysc.inverse_transform(model.predict(scaled.reshape(1,LOOKBACK,1),verbose=0).reshape(-1,1))[0,0])
        else: prediction = float(model.predict(np.asarray(values[-LOOKBACK:],dtype=float).reshape(1,-1))[0])
        prediction=max(0.,prediction)
        out.append(prediction);values.append(prediction)
    return np.asarray(out)

def run(depot, history, capacity_kw, epochs=12, min_targets=1000, source="historical", progress=None, training_config=None):
    """Train on one depot only; min_targets is an explicit user-controlled eligibility floor."""
    config = {"epochs": 12, "batch_size": 64, "learning_rate": 0.001,
              "lstm_units": 24, "lstm_layers": 1, "dropout": 0.0,
              "patience": 4, "optimizer": "Adam", "seed": 42,
              "rf_trees": 100, "rf_min_leaf": 2, "xgb_rounds": 160,
              "xgb_depth": 4, "xgb_learning_rate": 0.05}
    if training_config:
        unknown = set(training_config) - set(config)
        if unknown:
            raise ValueError(f"Unknown training settings: {sorted(unknown)}")
        config.update(training_config)
    config['epochs'] = int(epochs)
    for key, low, high in [('epochs', 4, 150), ('batch_size', 8, 512),
                           ('lstm_units', 8, 256), ('lstm_layers', 1, 3),
                           ('patience', 1, 30), ('seed', 0, 2147483647),
                           ('rf_trees', 10, 1000), ('rf_min_leaf', 1, 30),
                           ('xgb_rounds', 10, 1000), ('xgb_depth', 1, 12)]:
        config[key] = int(config[key])
        if not low <= config[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    for key, low, high in [('learning_rate', 1e-5, .1), ('dropout', 0, .6),
                           ('xgb_learning_rate', .001, .5)]:
        config[key] = float(config[key])
        if not np.isfinite(config[key]) or not low <= config[key] <= high:
            raise ValueError(f"{key} must be between {low} and {high}.")
    if config['optimizer'] not in ('Adam', 'RMSprop'):
        raise ValueError('Optimizer must be Adam or RMSprop.')
    min_targets = int(min_targets)
    if not 1000 <= min_targets <= 200000:
        raise ValueError("Minimum eligible targets must be between 1,000 and 200,000.")
    if source not in ('historical','synthetic'):
        raise ValueError('Invalid training data source.')
    series = synthetic_history(history) if source == 'synthetic' else pd.Series(history, dtype=float).sort_index()
    series.index = pd.to_datetime(series.index)
    series = series[~series.index.duplicated(keep='last')]
    def update(stage, fraction, detail):
        if progress is not None: progress(stage, fraction, detail)
    update('Preprocessing', .05, f'Loaded {len(series):,} {source} readings')
    report = inspect(series, min_targets)
    update('Preprocessing', .12, f"{report['eligible_targets']:,} eligible targets; {report['gaps']:,} timestamp gaps")
    # Do not bridge gaps: all input windows must contain consecutive 15-minute readings.
    values, times = series.to_numpy(), series.index
    features, targets, timestamps, prev_day = [], [], [], []
    for i in range(LOOKBACK, len(series)):
        if not np.isfinite(values[i-LOOKBACK:i+1]).all():
            continue
        if not np.all(np.diff(times[i-LOOKBACK:i+1].to_numpy(dtype="datetime64[ns]").astype("int64")) == 15*60*10**9):
            continue
        features.append(values[i-LOOKBACK:i])
        targets.append(values[i]); timestamps.append(times[i]); prev_day.append(values[i-LOOKBACK])
    if len(targets) < min_targets:
        raise ValueError(f"At least {min_targets:,} eligible consecutive-window targets are required; only {len(targets):,} were found.")
    update("Preprocessing", .22, "Creating chronological train, validation and test splits")
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    n = len(y); tr = int(n*.70); va = int(n*.85)
    if min(tr, va-tr, n-va) < 96:
        raise ValueError('Insufficient chronological training, validation or test targets.')
    xsc = StandardScaler().fit(X[:tr]); ysc = StandardScaler().fit(y[:tr,None])
    Xs = xsc.transform(X).astype('float32')
    ys = ysc.transform(y[:,None]).ravel().astype('float32')
    update("Baseline", .30, "Previous-day baseline evaluated; no epochs needed")
    predictions = {'Previous-day baseline':np.asarray(prev_day[va:],dtype=float)}
    durations = {'Previous-day baseline':0.0}
    start=time.perf_counter()
    rf=RandomForestRegressor(n_estimators=config["rf_trees"], min_samples_leaf=config["rf_min_leaf"], random_state=config["seed"], n_jobs=-1)
    update('Random Forest', .36, f'Fitting {config["rf_trees"]} trees')
    rf.fit(X[:tr], y[:tr]); update('Random Forest', .49, f'{config["rf_trees"]} trees fitted'); predictions['Random Forest']=rf.predict(X[va:]); durations['Random Forest']=time.perf_counter()-start
    start=time.perf_counter()
    xgb=XGBRegressor(n_estimators=config["xgb_rounds"],max_depth=config["xgb_depth"],learning_rate=config["xgb_learning_rate"],subsample=.9,colsample_bytree=.9,random_state=config["seed"],n_jobs=2)
    update('XGBoost', .53, f'Training {config["xgb_rounds"]} boosting rounds')
    xgb.fit(X[:tr],y[:tr],eval_set=[(X[tr:va],y[tr:va])],verbose=False)
    update('XGBoost', .66, f'{config["xgb_rounds"]} boosting rounds completed')
    predictions['XGBoost']=xgb.predict(X[va:]);durations['XGBoost']=time.perf_counter()-start
    keras.utils.set_random_seed(config['seed'])
    layers = [keras.layers.Input((LOOKBACK, 1))]
    for layer_index in range(config['lstm_layers']):
        layers.append(keras.layers.LSTM(config['lstm_units'],
                      return_sequences=layer_index < config['lstm_layers'] - 1))
        if config['dropout']:
            layers.append(keras.layers.Dropout(config['dropout']))
    layers.append(keras.layers.Dense(1))
    network = keras.Sequential(layers)
    optimizer_cls = keras.optimizers.Adam if config['optimizer'] == 'Adam' else keras.optimizers.RMSprop
    network.compile(optimizer=optimizer_cls(learning_rate=config['learning_rate']), loss='mse')
    start=time.perf_counter()
    class ProgressCallback(getattr(keras.callbacks, "Callback", object)):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            update('LSTM', .68 + .22*(epoch+1)/config["epochs"],
                   f"Epoch {epoch+1}/{int(epochs)} · loss {logs.get('loss',float('nan')):.4f} · validation {logs.get('val_loss',float('nan')):.4f}")
    fit=network.fit(Xs[:tr,:,None],ys[:tr],validation_data=(Xs[tr:va,:,None],ys[tr:va]),epochs=config["epochs"],batch_size=config["batch_size"],verbose=0,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=config['patience'],restore_best_weights=True),ProgressCallback()])
    predictions['LSTM']=ysc.inverse_transform(network.predict(Xs[va:,:,None],verbose=0)).ravel()
    durations['LSTM']=time.perf_counter()-start
    update('Evaluation', .92, 'Evaluating all four models on the same held-out observations')
    actual=y[va:].astype(float)
    rows=[]
    for name,pred in predictions.items():
        pred=np.asarray(pred,dtype=float)
        ab=actual>=capacity_kw;pb=pred>=capacity_kw
        tp=int((ab&pb).sum());fp=int((~ab&pb).sum());fn=int((ab&~pb).sum());tn=int((~ab&~pb).sum())
        recall=tp/(tp+fn) if tp+fn else np.nan
        precision=tp/(tp+fp) if tp+fp else np.nan
        specificity=tn/(tn+fp) if tn+fp else np.nan
        f1=2*precision*recall/(precision+recall) if np.isfinite(precision) and np.isfinite(recall) and precision+recall else np.nan
        rows.append({'Model':name,'MAE (kW)':mean_absolute_error(actual,pred),'RMSE (kW)':np.sqrt(mean_squared_error(actual,pred)), 'R²':r2_score(actual,pred), 'Breach recall (%)':recall*100,'False alarm ratio (%)':fp/(tp+fp)*100 if tp+fp else np.nan,'Breach precision (%)':precision*100,'F1-score (%)':f1*100,'Balanced accuracy (%)':(recall+specificity)*50 if np.isfinite(recall) and np.isfinite(specificity) else np.nan,'Specificity (%)':specificity*100,'Mean bias error (kW)':float(np.mean(pred-actual)), 'Training time (s)':durations[name]})
    frame=pd.DataFrame({'timestamp':pd.to_datetime(timestamps[va:]),'actual_kw':actual,**{name:np.asarray(v,dtype=float) for name,v in predictions.items()}})
    metrics=pd.DataFrame(rows)
    training=pd.DataFrame({'epoch':np.arange(1,len(fit.history['loss'])+1),'training_loss':fit.history['loss'],'validation_loss':fit.history['val_loss']})
    run_id = pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')
    metadata={'run_id':run_id,'depot':str(depot),'data_fingerprint':data_fingerprint(series),'capacity_kw':float(capacity_kw),'observations':int(n),'minimum_eligible_targets':min_targets,'train_targets':int(tr),'validation_targets':int(va-tr),'test_targets':int(n-va),'first_data_timestamp':str(times[0]),'last_data_timestamp':str(times[-1]),'train_start':str(timestamps[0]),'train_end':str(timestamps[tr-1]),'validation_start':str(timestamps[tr]),'validation_end':str(timestamps[va-1]),'test_start':str(timestamps[va]),'test_end':str(timestamps[-1]),'generated_at':pd.Timestamp.now(tz='UTC').isoformat(),'source':source,'lookback':LOOKBACK,'training_config':config,'epochs_completed':len(fit.history['loss']),'evaluation_type':'15-minute one-step-ahead held-out; not recursive multi-day validation'}
    p,m,pr,h,me=_paths(depot);p.mkdir(parents=True,exist_ok=True)
    # Keep each completed experiment in an immutable, depot-scoped version folder.
    # The root files remain the latest version for backwards-compatible loading.
    version_dir = p / 'versions' / run_id
    version_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(rf, version_dir / 'random_forest.joblib')
    if hasattr(xgb, 'save_model'): xgb.save_model(str(version_dir / 'xgboost.json'))
    if hasattr(network, 'save'): network.save(str(version_dir / 'lstm.keras'))
    joblib.dump(xsc, version_dir / 'x_scaler.joblib')
    joblib.dump(ysc, version_dir / 'y_scaler.joblib')
    metrics.to_csv(version_dir / 'metrics.csv',index=False)
    frame.to_csv(version_dir / 'predictions.csv',index=False)
    training.to_csv(version_dir / 'training_history.csv',index=False)
    (version_dir / 'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for src, dst in [(version_dir / 'metrics.csv',m),
                     (version_dir / 'predictions.csv',pr),
                     (version_dir / 'training_history.csv',h),
                     (version_dir / 'metadata.json',me)]:
        shutil.copyfile(src,dst)
    update('Completed', 1., 'Four models evaluated and versioned artifacts saved')
    return {'metrics':metrics,'predictions':frame,'history':training,'metadata':metadata}


def diagnose(history, min_targets=1000):
    """Auditable, non-destructive quality report for the selected input series."""
    raw = pd.Series(history)
    timestamps = pd.to_datetime(raw.index, errors='coerce')
    numeric = pd.to_numeric(raw, errors='coerce')
    valid_time = ~pd.isna(timestamps)
    work = pd.Series(numeric.to_numpy()[valid_time], index=pd.DatetimeIndex(timestamps[valid_time])).sort_index()
    duplicates = int(work.index.duplicated().sum())
    work = work[~work.index.duplicated(keep='last')]
    invalid = int((~np.isfinite(work.to_numpy(dtype=float))).sum())
    negative = int((work.to_numpy(dtype=float) < 0).sum())
    report = inspect(work, min_targets)
    gaps = work.index.to_series().diff().dropna()
    bad = gaps[gaps != pd.Timedelta(minutes=15)]
    largest = str(gaps.max()) if not gaps.empty else 'N/A'
    expected = int((work.index.max()-work.index.min())/pd.Timedelta(minutes=15))+1 if len(work)>1 else len(work)
    steps = [
        {'Step':'Input validation','Result':f'{len(raw):,} original rows; {int((~valid_time).sum()):,} invalid timestamps','Status':'Review' if not valid_time.all() else 'Pass'},
        {'Step':'Numeric demand validation','Result':f'{invalid:,} missing/non-finite values; {negative:,} negative readings','Status':'Review' if invalid or negative else 'Pass'},
        {'Step':'Duplicate timestamp check','Result':f'{duplicates:,} duplicate timestamps removed (keep last)','Status':'Review' if duplicates else 'Pass'},
        {'Step':'15-minute interval check','Result':f'{report["gaps"]:,} irregular transitions; {max(0,expected-len(work)):,} missing expected slots; largest interval {largest}','Status':'Review' if report['gaps'] else 'Pass'},
        {'Step':'Consecutive lookback windows','Result':f'{report["eligible_targets"]:,} eligible targets with {LOOKBACK} preceding intervals','Status':'Pass' if report['ready'] else 'Blocked'},
        {'Step':'Chronological split','Result':'70% train / 15% validation / 15% held-out test; at least 96 targets in each split','Status':'Ready' if report['ready'] else 'Not run'},
        {'Step':'Train-only scaling','Result':'StandardScaler fitted on training split only; validation and test are not used for fitting','Status':'Pending training'},
        {'Step':'Model training','Result':'Previous-day baseline, Random Forest (100 trees), XGBoost (160 rounds), LSTM (user-selected max epochs)','Status':'Pending training'},
        {'Step':'Held-out evaluation','Result':'MAE, RMSE, R², capacity-breach metrics and residual plots','Status':'Pending training'},
    ]
    recommendation = ('Enough consecutive readings are available to start training.' if report['ready'] else
                      f'Only {report["eligible_targets"]:,} eligible targets are available; {int(min_targets):,} required. '
                      'Inspect timestamp format and sampling frequency. Supply genuine consecutive 15-minute readings; '
                      'do not relabel or interpolate long gaps without documenting the method.')
    return {**report, 'steps':steps, 'recommendation':recommendation, 'duplicate_timestamps':duplicates,
            'invalid_values':invalid, 'negative_values':negative,
            'gap_examples':pd.DataFrame({'timestamp':bad.index.astype(str), 'interval':bad.astype(str).to_numpy()}).head(50)}


def export_bundle(result, depot):
    """Download all saved evidence as one self-contained ZIP file."""
    from io import BytesIO
    from zipfile import ZipFile, ZIP_DEFLATED
    buffer = BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
        for key, name in [('metrics','model_metrics.csv'), ('predictions','heldout_predictions.csv'),
                          ('history','lstm_epoch_history.csv')]:
            archive.writestr(name, result[key].to_csv(index=False))
        archive.writestr('experiment_metadata.json', json.dumps(result['metadata'], indent=2))
        archive.writestr('README.txt', f'Depot: {depot}\nOne-step-ahead held-out evaluation; not validated recursive forecasting.\nSynthetic training is for testing only.\n')
    return buffer.getvalue()

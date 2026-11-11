"""Optivolt depot updates and version management.

No TensorFlow import, model loading or retraining occurs when opening the page.
Model versions are tenant-scoped and activated only by an explicit user action.
"""
from __future__ import annotations
import hashlib
import html
import io
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
BUILTIN = {
    'Bexleyheath':17., 'Dartford':45., 'Islington':13., 'Mount Pleasant':260.,
    'Orpington':14., 'Premier Park':70., 'Whitechapel':55., 'Camden':21., 'Victoria':29.
}

def _scope():
    user = st.session_state.get('user') or {}
    raw = str(user.get('org_id') or 'unassigned')
    return re.sub(r'[^A-Za-z0-9_-]', '_', raw)[:80]

def _store():
    return Path(os.getenv('OPTIVOLT_STATE_DIR', '.app_state')).resolve() / 'organisations' / _scope() / 'depot_versions'

def _slug(name):
    return hashlib.sha256(name.casefold().strip().encode()).hexdigest()[:20]

def depot_dir(name):
    return _store() / _slug(name)

def registry_path():
    return _store() / 'registry.json'

def registry():
    p=registry_path()
    if not p.exists(): return {}
    try:
        d=json.loads(p.read_text())
        return d if isinstance(d,dict) else {}
    except (OSError,ValueError): return {}

def visible_depots():
    """Depots visible to the current organisation, including bundled UKPN depots."""
    records = registry()
    # Bundled research depots are explicitly opt-in demo data. Organisation
    # uploads remain tenant-scoped through registry() / _scope().
    demo = bool(st.session_state.get('optivolt_demo_mode', False))
    candidates = [*BUILTIN, *records] if demo else list(records)
    return [n for n in dict.fromkeys(candidates)
            if not records.get(n, {}).get('removed', False)]


def remove_depot(name):
    """Reversibly remove a depot from this organisation's workspace.

    Bundled UKPN source data and model files are never deleted. Uploaded data
    and model versions remain stored for restoration.
    """
    if name not in set(BUILTIN) | set(registry()):
        raise ValueError('Depot does not exist.')
    records = registry()
    rec = records.get(name, {})
    records[name] = {**rec, 'removed': True, 'removed_at': _utc()}
    _save_registry(records)


def restore_depot(name):
    records = registry()
    if name not in records or not records[name].get('removed'):
        raise ValueError('Depot is not archived.')
    records[name] = {**records[name], 'removed': False}
    _save_registry(records)


def render_depot_management():
    """Explicit, reversible depot removal with confirmation."""
    with st.expander('Manage depots · Remove or restore', expanded=False):
        st.caption('Remove hides a depot from this organisation’s workspace. '
                   'Its data and model versions are retained so it can be restored. '
                   'Bundled UKPN source files are never deleted.')
        names = visible_depots()
        if names:
            name = st.selectbox('Depot to remove', names, key='manage_remove_depot')
            st.caption('Original UKPN depot' if name in BUILTIN else 'Uploaded depot')
            confirmed = st.checkbox(f'Confirm removal of {name}', key=f'remove_confirm_{_slug(name)}')
            if st.button('Remove depot from workspace', disabled=not confirmed, key='remove_depot_action'):
                remove_depot(name)
                st.success(f'{name} removed from this workspace. You can restore it below.')
                st.rerun()
        else:
            st.info('No active depots in this workspace.')
        archived = [n for n, rec in registry().items() if rec.get('removed')]
        if archived:
            name = st.selectbox('Restore a removed depot', archived, key='manage_restore_depot')
            if st.button('Restore depot', key='restore_depot_action'):
                restore_depot(name)
                st.success(f'{name} restored.')
                st.rerun()


def _save_registry(d):
    path=registry_path();path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(d,indent=2),encoding='utf8')
    temp.replace(path)

def _builtin_file(name):
    return ROOT / f"UKPN_OptimisePrime_{name.replace(' ','_')}.csv"

def original_artifacts(name):
    slug=name.lower().replace(' ','_').replace('-','_')
    return ROOT/'models'/f'{slug}.keras',ROOT/'models'/f'{slug}_scalers.npz'

def active_paths(name):
    rec=registry().get(name,{})
    v=rec.get('active','original')
    if v!='original':
        folder=depot_dir(name)/'versions'/v
        m,s=folder/'model.keras',folder/'scalers.npz'
        if m.is_file() and s.is_file(): return m,s
    return original_artifacts(name)

def data_path(name):
    file=depot_dir(name)/'observations.csv'
    if file.is_file(): return file
    if name in BUILTIN: return _builtin_file(name)
    return None

def dataset_revision(name):
    p=data_path(name)
    return (str(p),p.stat().st_mtime_ns,p.stat().st_size) if p and p.exists() else ('',0,0)

def model_revision(name):
    m,s=active_paths(name)
    return (str(m),m.stat().st_mtime_ns if m.exists() else 0,str(s),s.stat().st_mtime_ns if s.exists() else 0)

@st.cache_data(show_spinner=False,max_entries=24)
def _read_csv(path,mtime,size):
    return pd.read_csv(path)

def observations(name):
    path=data_path(name)
    if path is None or not path.exists():
        raise FileNotFoundError(f'No saved observations available for {name}')
    frame=_read_csv(str(path),path.stat().st_mtime_ns,path.stat().st_size)
    return _normalise(frame)

def _normalise(df):
    df=df.copy(); df.columns=[str(c).strip() for c in df.columns]
    t=next((c for c in df if c.lower() in ('timestamp','datetime','date_time','time','date')),None)
    k=next((c for c in df if c.lower() in ('demand_kw','demand (kw)','demand','load_kw','power_kw','demandkw')),None)
    if t is None: t=next((c for c in df if 'time' in c.lower() or 'date' in c.lower()),None)
    if k is None: k=next((c for c in df if 'demand' in c.lower() or 'load' in c.lower() or 'kw' in c.lower()),None)
    if t is None or k is None: raise ValueError('Upload needs a timestamp/date column and demand (kW) column.')
    out=pd.DataFrame({'timestamp':pd.to_datetime(df[t],errors='coerce',utc=True).dt.tz_localize(None),
                      'demand_kw':pd.to_numeric(df[k],errors='coerce')}).dropna()
    out=out[(out.demand_kw>=0)&np.isfinite(out.demand_kw)]
    out=out.drop_duplicates('timestamp',keep='last').sort_values('timestamp')
    if out.empty: raise ValueError('No valid timestamps and nonnegative demand values found.')
    return out.reset_index(drop=True)

def preprocessing_audit(raw):
    """Report cleaning decisions on the raw uploaded file before normalization."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    t = next((c for c in df if c.lower() in ('timestamp','datetime','date_time','time','date')), None)
    k = next((c for c in df if c.lower() in ('demand_kw','demand (kw)','demand','load_kw','power_kw','demandkw')), None)
    if t is None: t = next((c for c in df if 'time' in c.lower() or 'date' in c.lower()), None)
    if k is None: k = next((c for c in df if 'demand' in c.lower() or 'load' in c.lower() or 'kw' in c.lower()), None)
    if t is None or k is None:
        raise ValueError('Upload needs a timestamp/date column and demand (kW) column.')
    times = pd.to_datetime(df[t], errors='coerce', utc=True).dt.tz_localize(None)
    demand = pd.to_numeric(df[k], errors='coerce')
    invalid_time = times.isna()
    invalid_demand = demand.isna() | ~np.isfinite(demand)
    negative = demand.lt(0) & ~invalid_demand
    eligible = ~(invalid_time | invalid_demand | negative)
    valid_times = times[eligible]
    duplicates = int(valid_times.duplicated(keep='last').sum())
    cleaned = _normalise(df)
    slot_counts = cleaned.timestamp.dt.floor('15min').value_counts()
    collided_slots = int((slot_counts - 1).clip(lower=0).sum())
    regular = _regular(cleaned)
    step = pd.Timedelta(minutes=15)
    gaps = regular.index.to_series().diff().dropna()
    missing_slots = int(sum(max(0, int(delta / step) - 1) for delta in gaps if delta > step))
    off_grid = int((cleaned.timestamp != cleaned.timestamp.dt.floor('15min')).sum())
    return [
        {'Step':'1. Read uploaded file','Result':f'{len(df):,} raw rows','Action':'Identify timestamp and demand columns'},
        {'Step':'2. Validate timestamps','Result':f'{int(invalid_time.sum()):,} invalid','Action':'Remove invalid or missing timestamps'},
        {'Step':'3. Validate demand values','Result':f'{int(invalid_demand.sum()):,} invalid','Action':'Remove missing, nonnumeric or infinite values'},
        {'Step':'4. Validate demand range','Result':f'{int(negative.sum()):,} negative','Action':'Remove negative kW readings'},
        {'Step':'5. Resolve duplicate timestamps','Result':f'{duplicates:,} duplicates','Action':'Keep last valid reading per exact timestamp'},
        {'Step':'6. Sort observations','Result':f'{len(cleaned):,} cleaned rows','Action':'Order chronologically'},
        {'Step':'7. Align 15-minute slots','Result':f'{off_grid:,} off-grid; {collided_slots:,} slot overlaps','Action':'Floor timestamps to 15-minute boundaries; average same-slot values'},
        {'Step':'8. Check continuity','Result':f'{missing_slots:,} missing 15-minute slots','Action':'Do not invent/interpolate missing readings; exclude windows spanning gaps'},
        {'Step':'9. Prepare model series','Result':f'{len(regular):,} usable time slots','Action':'Use chronological 70/15/15 split; training-only standardization'},
    ]


def read_upload_raw(blob, filename):
    if filename.lower().endswith('.csv'): return pd.read_csv(io.BytesIO(blob))
    if filename.lower().endswith('.xlsx'): return pd.read_excel(io.BytesIO(blob))
    raise ValueError('Upload CSV or XLSX only.')


@st.cache_data(show_spinner=False,max_entries=12)
def parse_upload(blob,filename):
    return _normalise(read_upload_raw(blob, filename))

def _utc():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def merge_data(name,new,mode='Append new timestamps'):
    if mode=='Replace dataset': return new.copy(),len(new),0
    try: old=observations(name)
    except FileNotFoundError: old=pd.DataFrame(columns=['timestamp','demand_kw'])
    if old.empty: return new.copy(),len(new),0
    new_only=new[~new.timestamp.isin(old.timestamp)]
    merged=pd.concat([old,new_only],ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return merged,len(new_only),len(new)-len(new_only)

def save_data(name,merged,capacity):
    folder=depot_dir(name);folder.mkdir(parents=True,exist_ok=True)
    p=folder/'observations.csv';tmp=folder/'observations.tmp'
    merged.to_csv(tmp,index=False);tmp.replace(p)
    reg=registry();prev=reg.get(name,{})
    reg[name]={**prev,'capacity_kw':float(capacity),'active':prev.get('active','original'),
               'last_data_update':_utc(),'rows':len(merged)}
    _save_registry(reg)
    _read_csv.clear()

def _load_model(name, version='active'):
    import tensorflow as tf
    if version=='original': paths=original_artifacts(name)
    elif version=='active': paths=active_paths(name)
    else:
        folder=depot_dir(name)/'versions'/version
        paths=(folder/'model.keras',folder/'scalers.npz')
    m,s=paths
    if not m.is_file() or not s.is_file(): raise FileNotFoundError(f'Model artifacts unavailable for {name}: {m.name}, {s.name}')
    model=tf.keras.models.load_model(str(m),compile=False)
    with np.load(s) as f:
        xm,xs,ym,ys=(float(np.asarray(f[k]).ravel()[0]) for k in ('x_mean','x_scale','y_mean','y_scale'))
    return model,(xm,max(xs,1e-12),ym,max(ys,1e-12))

def _windows(series,start=96):
    from numpy.lib.stride_tricks import sliding_window_view
    y=series.to_numpy(np.float32)
    if len(y)<=start:raise ValueError('At least 97 15-minute records are required.')
    windows=sliding_window_view(y,97)[:,:96].copy()
    # Compare timedeltas, not raw datetime integers: pandas can use ns, us or ms.
    # A 24-hour endpoint difference alone is not enough: check every 15-min step.
    index = pd.DatetimeIndex(series.index)
    steps_ok = (index[1:] - index[:-1]) == pd.Timedelta(minutes=15)
    consecutive = np.concatenate(([0], np.cumsum(~np.asarray(steps_ok, dtype=bool))))
    valid = (consecutive[96:] == consecutive[:-96])
    if not np.any(valid):
        longest = 1
        run = 1
        for good in steps_ok:
            run = run + 1 if good else 1
            longest = max(longest, run)
        raise ValueError(
            f'No 97-reading continuous window found. Longest consecutive run: {longest} readings '
            f'(required: 97). Check timestamp spacing and duplicates.'
        )
    return windows[valid],y[96:][valid],np.arange(96,len(y))[valid]

def _regular(df):
    # Group by 15-minute slots without filling months/years between uploads.
    aligned=df.assign(timestamp=df.timestamp.dt.floor('15min'))
    return aligned.groupby('timestamp').demand_kw.mean().sort_index().dropna()

def _predict(model,scalers,X):
    xm,xs,ym,ys=scalers
    scaled=((X-xm)/xs).astype(np.float32)[...,None]
    pred=model.predict(scaled,verbose=0,batch_size=512).reshape(-1)*ys+ym
    return np.maximum(pred,0.)

def _scores(actual,pred,capacity):
    err=actual-pred;breach=actual>=capacity;alert=pred>=capacity
    tp=int(np.sum(breach&alert));fn=int(np.sum(breach&~alert));fp=int(np.sum(~breach&alert))
    tn=int(np.sum(~breach&~alert))
    return {'MAE (kW)':round(float(np.mean(np.abs(err))),3),'RMSE (kW)':round(float(np.sqrt(np.mean(err**2))),3),
            'R²':round(float(1-np.sum(err**2)/np.sum((actual-actual.mean())**2)),3) if np.var(actual)>0 else None,
            'Breach recall (%)':round(100*tp/(tp+fn),1) if tp+fn else None,
            'False alarm ratio (%)':round(100*fp/(tp+fp),1) if tp+fp else None,
            'Breach precision (%)':round(100*tp/(tp+fp),1) if tp+fp else None,
            'F1-score':round(2*tp/(2*tp+fp+fn),3) if 2*tp+fp+fn else None,
            'Balanced accuracy (%)':round(50*((tp/(tp+fn))+(tn/(tn+fp))),1) if tp+fn and tn+fp else None,
            'Specificity (%)':round(100*tn/(tn+fp),1) if tn+fp else None,
            'Mean bias error (kW)':round(float(np.mean(pred-actual)),3),
            'Observations':int(len(actual))}

def evaluate(name,new,capacity,version='active'):
    # Upload must be later than saved historical data to support genuinely new-data scoring.
    series=_regular(new)
    X,y,_=_windows(series)
    model,scalers=_load_model(name,version)
    scores=_scores(y,_predict(model,scalers,X),capacity)
    del model
    return scores

MIN_TRAINING_OBSERVATIONS = 1000
EVALUATION_METRICS = (
    'MAE (kW)', 'RMSE (kW)', 'R²', 'Breach recall (%)',
    'False alarm ratio (%)', 'Breach precision (%)', 'F1-score',
    'Balanced accuracy (%)', 'Specificity (%)', 'Mean bias error (kW)',
)


def _selected_metric_columns():
    selected = st.multiselect(
        'Evaluation metrics to display', EVALUATION_METRICS,
        default=list(EVALUATION_METRICS[:5]), key='optivolt_selected_evaluation_metrics',
        help='The original five metrics are selected by default. Choose any of the ten.'
    )
    return selected or list(EVALUATION_METRICS[:5])


def _uploaded_model_decision(model_scores, baseline_scores):
    """Conservative recommendation based on a matched previous-day baseline."""
    if not baseline_scores or model_scores.get('Observations', 0) < 96:
        return ('Further validation required',
                'Insufficient comparable observations or baseline evidence. Do not approve this model yet.')
    mae = model_scores.get('MAE (kW)')
    base_mae = baseline_scores.get('MAE (kW)')
    if mae is None or base_mae is None:
        return 'Further validation required', 'MAE comparison is unavailable.'
    recall = model_scores.get('Breach recall (%)')
    if mae >= base_mae:
        return ('Retraining recommended',
                f'Model MAE ({mae:.3f} kW) does not beat the previous-day baseline ({base_mae:.3f} kW). '
                'Review data quality and train a depot-specific model.')
    if recall is not None and recall < 80:
        return ('Retraining / further validation recommended',
                f'Model beats the baseline on MAE ({mae:.3f} vs {base_mae:.3f} kW), '
                f'but breach recall is only {recall:.1f}%. Review missed peaks before using it for planning.')
    if recall is None:
        return ('Promising — breach detection not assessed',
                f'Model MAE ({mae:.3f} kW) beats baseline ({base_mae:.3f} kW), '
                'but no actual breaches were available to measure recall. Test on representative peaks.')
    return ('Promising — human review required',
            f'Model beats previous-day baseline on MAE ({mae:.3f} vs {base_mae:.3f} kW) '
            f'and breach recall is {recall:.1f}%. Review false alarms and multi-day forecasts before activation.')


def _evaluate_with_baseline(name, incoming, capacity):
    """Evaluate saved model and previous-day baseline on identical eligible targets."""
    series = _regular(incoming)
    X, y, positions = _windows(series)
    model, scalers = _load_model(name)
    prediction = _predict(model, scalers, X)
    del model
    model_scores = _scores(y, prediction, capacity)
    # A previous-day baseline needs a reading exactly 96 15-minute slots earlier.
    timestamps = series.index
    prior = timestamps[positions] - pd.Timedelta(days=1)
    prior_values = series.reindex(prior).to_numpy(dtype=float)
    valid = np.isfinite(prior_values)
    baseline_scores = _scores(y[valid], prior_values[valid], capacity) if valid.any() else None
    # Compare model on exactly the baseline-eligible observations, not different samples.
    matched_scores = _scores(y[valid], prediction[valid], capacity) if valid.any() else None
    return model_scores, baseline_scores, matched_scores



def training_readiness(df, minimum_training_rows=MIN_TRAINING_OBSERVATIONS):
    """Count eligible training targets, not merely uploaded CSV rows."""
    series = _regular(df)
    n = len(series)
    cut = int(n * .70)
    val_end = int(n * .85)
    try:
        _, _, target_positions = _windows(series)
        train_count = int(np.count_nonzero(target_positions < cut))
        validation_count = int(np.count_nonzero((target_positions >= cut) & (target_positions < val_end)))
        test_count = int(np.count_nonzero(target_positions >= val_end))
        detail = None
    except ValueError as exc:
        train_count = validation_count = test_count = 0
        detail = str(exc)
    ready = train_count >= minimum_training_rows and validation_count >= 96 and test_count >= 96
    return {'valid_rows': n, 'train_rows': train_count, 'validation_rows': validation_count,
            'test_rows': test_count, 'ready': ready, 'detail': detail}


def train_candidate(name,merged,capacity,epochs,progress,batch_size=128,learning_rate=0.001,patience=4,status=None,minimum_training_rows=MIN_TRAINING_OBSERVATIONS):
    import tensorflow as tf
    if status: status.info('Stage 1/6 · Cleaning and preprocessing verified: invalid values removed, 15-minute slots aligned, gaps excluded from windows')
    readiness = training_readiness(merged, minimum_training_rows)
    if not readiness['ready']:
        raise ValueError(
            f"Training requires at least {minimum_training_rows:,} eligible training observations "
            f"after the chronological split and 96-reading windows; available: "
            f"{readiness['train_rows']:,}. Validation and test must each have at least 96 targets. "
            f"{readiness['detail'] or ''}"
        )
    s=_regular(merged)
    arr=s.to_numpy(np.float32)
    n=len(arr);cut=int(n*.70);val_end=int(n*.85)
    if cut<192 or val_end-cut<96 or n-val_end<96:
        raise ValueError('Need enough observations for 70% train / 15% validation / 15% test.')
    if status: status.info('Stage 2/6 · Chronological 70/15/15 split, continuous 96-reading windows and training-only scaling')
    progress.progress(0.10, text='Preparing data and fitting training-only scaler')
    # Refit scaler on TRAINING DATA ONLY to prevent future leakage.
    xm=float(arr[:cut].mean());xs=max(float(arr[:cut].std()),1e-12)
    X,y,starts=_windows(s)
    tr=starts<cut;va=(starts>=cut)&(starts<val_end);te=starts>=val_end
    x=((X-xm)/xs)[...,None].astype(np.float32)
    target=((y-xm)/xs).astype(np.float32)
    # Same single-step input/output LSTM family as original. Train candidate from scratch.
    model=tf.keras.Sequential([tf.keras.layers.Input((96,1)),tf.keras.layers.LSTM(64,return_sequences=True),
        tf.keras.layers.LSTM(32),tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss='mse')
    if status: status.info('Stage 3/6 · Training the new LSTM; validation loss is checked after every epoch')
    progress.progress(0.20, text='Training started · epoch 0')
    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            logs=logs or {}
            progress.progress(min(0.75,0.20+0.55*(epoch+1)/epochs),text=f'Epoch {epoch+1}/{epochs} · training loss {logs.get("loss",float("nan")):.4f} · validation loss {logs.get("val_loss",float("nan")):.4f}')
    model.fit(x[tr],target[tr],validation_data=(x[va],target[va]),epochs=epochs,
        batch_size=batch_size,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True),Callback()])
    if status: status.info('Stage 4/6 · Evaluating on held-out timestamps not used to fit the new model')
    progress.progress(0.79, text='Calculating held-out accuracy and previous-day baseline')
    scaler=(xm,xs,xm,xs)
    test_actual=y[te].astype(float); candidate_pred=_predict(model,scaler,X[te])
    candidate_scores=_scores(test_actual,candidate_pred,capacity)
    baseline_predictions=[]; baseline_actual=[]
    for idx, observed in zip(starts[te], test_actual):
        if idx >= 96 and (s.index[idx] - s.index[idx-96]) == pd.Timedelta(days=1):
            baseline_predictions.append(float(arr[idx-96]));baseline_actual.append(float(observed))
    baseline_scores=_scores(np.asarray(baseline_actual),np.asarray(baseline_predictions),capacity) if baseline_actual else None
    if status: status.info('Stage 5/6 · Comparing new model, existing active model and previous-day baseline')
    progress.progress(0.88, text='Comparing model accuracy on matching held-out observations')
    old_scores=None
    try:
        old_model,old_scalers=_load_model(name)
        old_scores=_scores(test_actual,_predict(old_model,old_scalers,X[te]),capacity)
        del old_model
    except FileNotFoundError:pass
    stamp=_utc();existing=[v for v in available_versions(name) if re.fullmatch(r'v\d+',v)]
    next_number=max([int(v[1:]) for v in existing],default=0)+1
    version=f'v{next_number}'
    folder=depot_dir(name)/'versions'/version;folder.mkdir(parents=True,exist_ok=False)
    model.save(str(folder/'model.keras'))
    np.savez_compressed(folder/'scalers.npz',x_mean=np.array([xm]),x_scale=np.array([xs]),
                        y_mean=np.array([xm]),y_scale=np.array([xs]))
    meta={'depot':name,'version':version,'created':stamp,'train_rows':int(tr.sum()),'minimum_training_rows':minimum_training_rows,'total_valid_rows':int(n),'validation_rows':int(va.sum()),
          'heldout_rows':int(te.sum()),'data_through':str(s.index.max()),'candidate_metrics':candidate_scores,
          'previous_active_metrics':old_scores,'previous_day_baseline_metrics':baseline_scores,'trained_on':'70% chronological training; 15% validation; last 15% held out'}
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    if status: status.success('Stage 6/6 · New model saved as '+version+'; it is not active until you choose Activate')
    progress.progress(1.0,text='Training, evaluation and version save complete')
    del model
    return meta

def activate(name,version):
    if version!='original':
        p=depot_dir(name)/'versions'/version
        if not (p/'model.keras').exists() or not (p/'scalers.npz').exists():raise ValueError('Candidate files not found')
    elif name not in BUILTIN:raise ValueError('A new depot has no bundled original model.')
    d=registry();record=d.get(name,{})
    record['active']=version;record['activated_at']=_utc();d[name]=record
    _save_registry(d)

def available_versions(name):
    result=['original'] if name in BUILTIN else []
    folder=depot_dir(name)/'versions'
    if folder.exists(): result += sorted((p.name for p in folder.iterdir() if p.is_dir() and (p/'model.keras').exists()),reverse=True)
    return result

def version_export(name,version):
    if version=='original':m,s=original_artifacts(name);meta=None
    else:
        d=depot_dir(name)/'versions'/version;m,s,meta=d/'model.keras',d/'scalers.npz',d/'metadata.json'
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(m,arcname='model.keras');z.write(s,arcname='scalers.npz')
        if meta is not None and meta.exists():z.write(meta,arcname='metadata.json')
        p=data_path(name)
        if p and p.is_file(): z.write(p,arcname='observations.csv')
    return out.getvalue()

def _compatibility_report(model_name, uploaded):
    """Show explicit data and model checks before allowing evaluation."""
    checks=[]
    def add(item, ok, detail):
        checks.append({'Compatibility criterion':item,'Result':'PASS' if ok else 'FAIL','Details':detail})
    required={'timestamp','demand_kw'}
    schema=required.issubset(uploaded.columns)
    add('Required data columns',schema,'timestamp and demand_kw required')
    if not schema:
        return False,checks
    timestamps=pd.to_datetime(uploaded.timestamp,errors='coerce')
    values=pd.to_numeric(uploaded.demand_kw,errors='coerce')
    valid=timestamps.notna() & values.notna() & np.isfinite(values) & (values>=0)
    add('Valid timestamps and non-negative demand',bool(valid.all()),f'{int(valid.sum()):,} of {len(uploaded):,} valid rows')
    aligned=bool(((timestamps.dropna().dt.minute % 15)==0).all() and (timestamps.dropna().dt.second==0).all())
    add('15-minute timestamp alignment',aligned,'All timestamps must fall on a 15-minute boundary')
    duplicate=bool(timestamps.dropna().duplicated().any())
    add('No duplicate timestamps',not duplicate,'Duplicate readings must be resolved before evaluation' if duplicate else 'No duplicates detected')
    try:
        X,y,_=_windows(_regular(uploaded))
        add('Continuous 96-step input and target',True,f'{len(y):,} valid evaluation windows of 97 consecutive readings')
    except Exception as exc:
        add('Continuous 96-step input and target',False,str(exc))
    paths=active_paths(model_name)
    exists=all(path.is_file() for path in paths)
    add('Saved model and scaler files',exists,f'{model_name} model artifacts '+('found' if exists else 'not found'))
    if exists:
        try:
            model,scalers=_load_model(model_name)
            shape=model.input_shape
            if isinstance(shape,list):shape=shape[0]
            shape_ok=len(shape)==3 and shape[1]==96 and shape[2]==1
            add('LSTM input shape',shape_ok,f'Model expects {shape}; required (batch, 96, 1)')
            scaler_ok=bool(np.all(np.isfinite(scalers)) and scalers[1]>0 and scalers[3]>0)
            add('Scaler parameters',scaler_ok,'Finite means and positive scales required')
            del model
        except Exception as exc:
            add('Model loading',False,f'Could not inspect saved model: {exc}')
    return all(item['Result']=='PASS' for item in checks),checks


def _model_compatibility(name,uploaded):
    ok,checks=_compatibility_report(name,uploaded)
    return ok,[row['Compatibility criterion']+': '+row['Details'] for row in checks if row['Result']=='FAIL']

def _model_decision(new,previous,baseline):
    """Decision support, not an unsupported accuracy certification."""
    n=float(new['MAE (kW)'])
    comparisons=[]
    if previous and previous.get('MAE (kW)') is not None:
        comparisons.append(('existing active model',float(previous['MAE (kW)'])))
    if baseline and baseline.get('MAE (kW)') is not None:
        comparisons.append(('previous-day baseline',float(baseline['MAE (kW)'])))
    if not comparisons:
        return 'Further validation required', 'No comparable existing-model or previous-day baseline result is available. Review the held-out predictions before activation.'
    failed=[label for label,value in comparisons if n>=value]
    if failed:
        return 'Keep existing model / improve new model', 'The new model does not outperform '+', '.join(failed)+' on held-out MAE. Check data quality, add representative observations, tune epochs and learning rate, and retrain before activation.'
    return 'New model is promising — review before activation', 'The new model has lower held-out MAE than '+ ' and '.join(label for label,_ in comparisons)+'. Review RMSE, peak-event detection and forecast plots before choosing Activate. One-step evaluation does not validate multi-day recursive forecasts.'


def _version_label(version):
    return 'Original saved model' if version=='original' else ('New model '+version.upper() if re.fullmatch(r'v\d+',version) else 'Legacy model · '+version)

def render_setup(selected=None):
    st.title('Depot Setup')
    st.caption('Manage depot data, grid limits and saved LSTM models in one workspace.')
    tab_config, tab_data = st.tabs(['Grid Capacity & Thresholds', 'Add / Update Depot Data & Model'])
    with tab_data:
        st.markdown('### Upload, validate and train')
        st.caption('Upload 15-minute demand data, inspect validation results, evaluate a saved model or explicitly train a new candidate. Training progress appears here; uploading alone never retrains the model.')
        render_upload(embedded=True)
    with tab_config:
        _render_capacity_setup(selected)


def _render_capacity_setup(selected=None):
    names=visible_depots()
    if not names:
        st.info('No depot has been added yet. Select the Add / Update Depot Data & Model tab above to upload and validate your first depot dataset.')
        return
    default=names.index(selected) if selected in names else 0
    name=st.selectbox('Depot to configure',names,index=default,key='fast_setup_depot')
    info=registry().get(name,{})
    overrides=st.session_state.get('depot_threshold_overrides',{})
    ov=overrides.get(name,{})
    cap=float(info.get('capacity_kw',ov.get('capacity_kw',BUILTIN.get(name,17.))))
    cap_value=st.number_input('Grid capacity (kW)',min_value=0.1,value=cap,step=1.,key=f'cfg_cap_{_slug(name)}')
    warning=st.number_input('Warning threshold (kW)',min_value=0.,value=float(ov.get('warning_kw',info.get('warning_kw',round(cap*.8,1)))),step=1.,key=f'cfg_warn_{_slug(name)}')
    if st.button('Save depot configuration',type='primary'):
        if warning>cap_value:st.error('Warning threshold cannot exceed grid capacity.')
        else:
            overrides=dict(overrides);overrides[name]={'capacity_kw':cap_value,'warning_kw':warning}
            st.session_state['depot_threshold_overrides']=overrides
            reg=registry()
            if name in reg:
                reg[name]={**reg[name],'capacity_kw':cap_value,'warning_kw':warning};_save_registry(reg)
            st.session_state['grid_limit']=cap_value
            st.session_state['warn_threshold']=warning
            st.success(f'Saved grid capacity for {name} in this deployment.')

def render_upload(default_mode=None, embedded=False):
    if not embedded:
        st.title('Upload & Update Depot Data')
    if not embedded:
        st.caption('Add a new depot, append observations to an existing depot, evaluate the active model, and train a candidate. Your original model is never overwritten.')
    st.markdown("""<style>
    /* Explicit card typography: override app-level -webkit-text-fill-color rules. */
    div.st-key-ov_upload_summary_cards .ov-upload-kpi,
    div.st-key-ov_upload_summary_cards .ov-upload-kpi *,
    .ov-upload-kpi, .ov-upload-kpi * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }
    div.st-key-ov_upload_summary_cards .ov-upload-kpi {
        background: #17365D !important;
        border: 2px solid #3478F6 !important;
        border-top-width: 5px !important;
        border-radius: 16px !important;
        min-height: 132px;
        padding: 18px;
    }
    /* Save and evaluate actions: consistent contrast for all button types. */
    div.st-key-ov_depot_card_upload [data-testid="stButton"] button:not(:disabled) {
        background: #2563EB !important;
        border: 1px solid #93B4FF !important;
        color: #FFFFFF !important;
    }
    div.st-key-ov_depot_card_upload [data-testid="stButton"] button:not(:disabled) * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }
    div.st-key-ov_depot_card_upload [data-testid="stButton"] button:not(:disabled):hover {
        background: #1D4ED8 !important;
    }
    div.st-key-ov_depot_card_upload [data-testid="stButton"] button:disabled,
    div.st-key-ov_depot_card_upload [data-testid="stButton"] button:disabled * {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
        opacity: 1 !important;
    }
    /* Expander summary and its arrow remain legible even with global light-theme CSS. */
    div.st-key-ov_depot_card_upload [data-testid="stExpander"] summary,
    div.st-key-ov_depot_card_upload [data-testid="stExpander"] summary *,
    div.st-key-ov_depot_card_manage [data-testid="stExpander"] summary,
    div.st-key-ov_depot_card_manage [data-testid="stExpander"] summary * {
        background: #17365D !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }
    /* Upload save/evaluate buttons: opaque readable text, including disabled state. */
    div.st-key-ov_depot_card_upload [data-testid="stButton"] button[kind="primary"] {
        background: #2563EB !important; color: #FFFFFF !important;
        border: 1px solid #93B4FF !important;
    }
    div.st-key-ov_depot_card_upload [data-testid="stButton"] button[kind="primary"] * {
        color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
    }
    div.st-key-ov_depot_card_upload [data-testid="stButton"] button:disabled {
        background: #E2E8F0 !important; color: #334155 !important;
        opacity: 1 !important;
    }
    div.st-key-ov_depot_card_upload [data-testid="stButton"] button:disabled * {
        color: #334155 !important; -webkit-text-fill-color: #334155 !important;
    }
    /* Depot setup: readable accordion and metric surfaces. */
    div[data-testid="stExpander"]:has(summary) {
        border: 1px solid #3478F6 !important;
        background: #17365D !important;
        border-radius: 16px !important;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary *,
    div[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #17365D !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] label,
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"] {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] [data-baseweb="select"] *,
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] input {
        color: #112B4A !important;
        -webkit-text-fill-color: #112B4A !important;
    }
    /* Metrics elsewhere in Depot Setup: prevent white-on-light cards. */
    div[data-testid="stMetric"] {
        background: #17365D !important;
        background-image: none !important;
        border: 1px solid #3478F6 !important;
        border-top: 4px solid #3478F6 !important;
        border-radius: 16px !important;
        padding: 16px 18px !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
    div[data-testid="stMetric"] [data-testid="stMetricDelta"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }
    /* Training readiness: navy cards and opaque white metrics, scoped to these four cards only. */
    div.st-key-ov_training_readiness_metrics [data-testid="stMetric"] {
        background: #17365D !important;
        background-image: none !important;
        border: 1px solid #3478F6 !important;
        border-top: 4px solid #3478F6 !important;
        border-radius: 16px !important;
        padding: 16px 18px !important;
        min-height: 142px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, .12);
    }
    div.st-key-ov_training_readiness_metrics [data-testid="stMetric"] [data-testid="stMetricLabel"],
    div.st-key-ov_training_readiness_metrics [data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div.st-key-ov_training_readiness_metrics [data-testid="stMetric"] [data-testid="stMetricValue"],
    div.st-key-ov_training_readiness_metrics [data-testid="stMetric"] [data-testid="stMetricValue"] *,
    div.st-key-ov_training_readiness_metrics [data-testid="stMetric"] [data-testid="stMetricDelta"],
    div.st-key-ov_training_readiness_metrics [data-testid="stMetric"] [data-testid="stMetricDelta"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }
    div.st-key-ov_training_readiness_metrics [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 750 !important;
    }
    /* Scoped to the Depot Setup workflow cards; controls remain native Streamlit widgets. */
    div.st-key-ov_depot_card_select, div.st-key-ov_depot_card_upload,
    div.st-key-ov_depot_card_models, div.st-key-ov_depot_card_manage {
        background: #102b50 !important; border: 1px solid #2b527d !important;
        border-radius: 16px !important; padding: 20px !important;
        margin: 12px 0 18px !important;
    }
    div.st-key-ov_depot_card_select label, div.st-key-ov_depot_card_upload label,
    div.st-key-ov_depot_card_models label, div.st-key-ov_depot_card_manage label,
    div.st-key-ov_depot_card_select p, div.st-key-ov_depot_card_upload p,
    div.st-key-ov_depot_card_models p, div.st-key-ov_depot_card_manage p,
    div.st-key-ov_depot_card_select h3, div.st-key-ov_depot_card_upload h3,
    div.st-key-ov_depot_card_models h3, div.st-key-ov_depot_card_manage h3,
    div.st-key-ov_depot_card_select h4, div.st-key-ov_depot_card_upload h4,
    div.st-key-ov_depot_card_models h4, div.st-key-ov_depot_card_manage h4 {
        color: #f4f8ff !important;
    }
    div.st-key-ov_depot_card_select [data-testid="stCaptionContainer"] p,
    div.st-key-ov_depot_card_upload [data-testid="stCaptionContainer"] p,
    div.st-key-ov_depot_card_models [data-testid="stCaptionContainer"] p {
        color: #d0e2f2 !important;
    }
    div.st-key-ov_depot_card_upload [data-testid="stMetric"] {
        background: #21466f; border-radius: 12px; padding: 12px;
    }
    div.st-key-ov_depot_card_select [data-testid="stWidgetLabel"] *,
    div.st-key-ov_depot_card_upload [data-testid="stWidgetLabel"] *,
    div.st-key-ov_depot_card_models [data-testid="stWidgetLabel"] *,
    div.st-key-ov_depot_card_select [data-testid="stCheckbox"] *,
    div.st-key-ov_depot_card_upload [data-testid="stCheckbox"] * {color:#ffffff !important;opacity:1 !important;}
    div.st-key-ov_depot_card_upload [data-testid="stMetricValue"],
    div.st-key-ov_depot_card_upload [data-testid="stMetricLabel"] {
        color: #fff !important;
    }
    /* High contrast for every label and message inside dark workflow cards. */
    div[class*="st-key-ov_depot_card_"] [data-testid="stWidgetLabel"],
    div[class*="st-key-ov_depot_card_"] [data-testid="stWidgetLabel"] *,
    div[class*="st-key-ov_depot_card_"] [data-testid="stMarkdownContainer"] > p,
    div[class*="st-key-ov_depot_card_"] [data-testid="stMarkdownContainer"] > p *,
    div[class*="st-key-ov_depot_card_"] [data-testid="stMarkdownContainer"] > h3,
    div[class*="st-key-ov_depot_card_"] [data-testid="stMarkdownContainer"] > h4,
    div[class*="st-key-ov_depot_card_"] [data-testid="stCheckbox"] label span,
    div[class*="st-key-ov_depot_card_"] [data-testid="stRadio"] label span,
    div[class*="st-key-ov_depot_card_"] [data-testid="stCaptionContainer"] p {
        color: #f8fbff !important;
        -webkit-text-fill-color: #f8fbff !important;
        opacity: 1 !important;
    }
    /* Audit heading and caption: solid white on the navy upload card. */
    div.st-key-ov_depot_card_upload [data-testid="stMarkdownContainer"] h5,
    div.st-key-ov_depot_card_upload [data-testid="stMarkdownContainer"] h5 *,
    div.st-key-ov_depot_card_upload [data-testid="stCaptionContainer"],
    div.st-key-ov_depot_card_upload [data-testid="stCaptionContainer"] p,
    div.st-key-ov_depot_card_upload [data-testid="stCaptionContainer"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }
    /* Light input surfaces must retain dark input text. */
    div[class*="st-key-ov_depot_card_"] input,
    div[class*="st-key-ov_depot_card_"] textarea,
    div[class*="st-key-ov_depot_card_"] [data-baseweb="select"] *,
    div[class*="st-key-ov_depot_card_"] [role="combobox"] {
        color: #14213d !important;
        -webkit-text-fill-color: #14213d !important;
    }
    div[class*="st-key-ov_depot_card_"] [data-testid="stAlert"] *,
    div[class*="st-key-ov_depot_card_"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        -webkit-text-fill-color: initial;
    }
    </style>""", unsafe_allow_html=True)
    with st.container(key="ov_depot_card_manage"):
        st.markdown("#### Depot management")
        render_depot_management()
    with st.container(key="ov_depot_card_select"):
        st.markdown("#### Step 1 — Select or create a depot")
        names=visible_depots()
        mode = default_mode if default_mode in ('Update existing depot', 'Add new depot') else st.radio('What would you like to do?',('Update existing depot','Add new depot'),horizontal=True)
        if mode=='Update existing depot':
            if not names:
                st.info('No depots available. Add a new depot or restore an archived one.')
                return
            name=st.selectbox('Existing depot',names,key='update_existing_name')
        else:
            name=st.text_input('New depot name',key='new_depot_name').strip()
            if name in names: st.warning('This depot exists. Choose Update existing depot.')
        if not name:return
        info=registry().get(name,{})
        capacity=st.number_input('Grid capacity (kW)',min_value=.1,value=float(info.get('capacity_kw',BUILTIN.get(name,17.))),step=1.)
    with st.container(key="ov_depot_card_upload"):
        st.markdown('#### Step 2 — Upload and validate observations')
        file=st.file_uploader('Upload 15-minute demand observations (CSV or Excel)',type=['csv','xlsx'],key='model_update_file')
        incoming=None;merged=None
        if file is not None:
            try:
                blob=file.getvalue(); raw_upload=read_upload_raw(blob,file.name)
                incoming=parse_upload(blob,file.name)
                st.markdown('##### Data cleaning and preprocessing — audit')
                st.caption('These steps are applied before training. Counts refer to the uploaded file, not the entire merged depot history.')
                audit=preprocessing_audit(raw_upload)
                st.dataframe(pd.DataFrame(audit),hide_index=True,use_container_width=True)
                st.success(f"Preprocessing complete: {len(raw_upload):,} raw readings → {len(incoming):,} cleaned readings.")
                with st.expander('How preprocessing protects the LSTM'):
                    st.markdown('**Continuity:** Only 96-reading inputs with a following target at consecutive 15-minute intervals are used. Missing time slots are not interpolated.\n\n**Splitting:** The data is divided chronologically into 70% training, 15% validation and 15% held-out testing.\n\n**Scaling:** Mean and standard deviation are calculated from the training period only and then applied to validation/test readings to avoid future-data leakage.\n\n**Model training:** The LSTM is trained from scratch only after you select **Train new LSTM model**. Saved models are unchanged until you explicitly activate a new version.')
                st.markdown('**Choose how to save this uploaded file**')
                st.info('To ADD readings: select the same depot, upload a second CSV with later 15-minute timestamps, choose Add, then click the blue Save button below. To REPLACE readings: upload a complete replacement CSV, choose Replace, then click Save. Selecting a radio option alone does not save anything.')
                try:
                    existing_obs = observations(name)
                except FileNotFoundError:
                    existing_obs = pd.DataFrame(columns=['timestamp', 'demand_kw'])
                if not existing_obs.empty:
                    st.caption(f"Currently saved for {name}: {len(existing_obs):,} readings. Latest saved timestamp: {pd.to_datetime(existing_obs.timestamp).max():%Y-%m-%d %H:%M}.")
                else:
                    st.caption(f'No readings are currently saved for {name}. The uploaded file will create the dataset.')
                mode_merge_label=st.radio('How to use these observations?',('Add new readings to existing data','Replace all existing readings'),horizontal=True)
                mode_merge='Append new timestamps' if mode_merge_label=='Add new readings to existing data' else 'Replace dataset'
                merged,new_rows,overlaps=merge_data(name,incoming,mode_merge)
                # Render explicit high-contrast HTML cards instead of st.metric; app-wide
                # theme rules otherwise paint the metric container light and its text white.
                with st.container(key='ov_upload_summary_cards'):
                    a,b,c=st.columns(3)
                    for col,label,value in (
                        (a,'Uploaded valid rows',len(incoming)),
                        (b,'New timestamps to add' if mode_merge=='Append new timestamps' else 'Replacement readings',new_rows),
                        (c,'Total saved rows after update',len(merged)),
                    ):
                        with col:
                            st.markdown(
                                '<div class="ov-upload-kpi" style="background:#17365D!important;color:#FFFFFF!important;'
                                '-webkit-text-fill-color:#FFFFFF!important;'
                                'border:2px solid #3478F6;border-top:5px solid #3478F6;'
                                'border-radius:16px;padding:18px;min-height:132px;">'
                                f'<div style="color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;font-size:15px;font-weight:650;">{html.escape(label)}</div>'
                                f'<div style="color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;font-size:36px;font-weight:800;margin-top:10px;">{value:,}</div>'
                                '</div>',unsafe_allow_html=True)
                st.caption(f'{incoming.timestamp.min():%d %b %Y} to {incoming.timestamp.max():%d %b %Y}. Existing timestamps skipped: {overlaps}. Uploading does not retrain automatically.')
                if overlaps and mode_merge=='Append new timestamps':
                    st.info(f'{overlaps:,} uploaded timestamps already exist and will be skipped; existing values will be retained.')
                if mode_merge=='Replace dataset' and not existing_obs.empty:
                    st.warning('Replace will overwrite the saved observation dataset for this depot. Saved LSTM model versions will not be retrained.')
                st.markdown('### Final step — save the uploaded readings')
                st.caption('The numbers above are a preview. Your depot data changes only when you click Save.')
                if new_rows == 0 and mode_merge == 'Append new timestamps':
                    st.info('Nothing new to save: all uploaded timestamps are already stored. Upload readings with later or otherwise missing 15-minute timestamps to append data.')
                if st.button('Save uploaded readings to this depot',type='primary',key='save_uploaded_observations',
                             disabled=(new_rows==0 and mode_merge=='Append new timestamps'),use_container_width=True):
                    save_data(name,merged,capacity)
                    st.success(f'SAVED TO DEPOT — {new_rows:,} readings added or replaced; {len(merged):,} readings now stored for {name}. No model was retrained.')
                st.markdown('#### Step 3 — Select and verify a saved LSTM model')
                st.caption('Bexleyheath is the default reference model. You can select any of the nine UKPN demonstration models. Model compatibility is verified against the uploaded observations before evaluation.')
                model_names=list(BUILTIN)
                chosen_model=st.selectbox('Saved LSTM model to evaluate',model_names,index=model_names.index('Bexleyheath'),key='compat_model_selection')
                compatible,checks=_compatibility_report(chosen_model,incoming)
                st.markdown('**Compatibility criteria — results**')
                st.dataframe(pd.DataFrame(checks),hide_index=True,use_container_width=True)
                if compatible:
                    st.success(f'PASS — {chosen_model} model is technically compatible with the uploaded data. You can evaluate its predictive accuracy below.')
                    if chosen_model != name:
                        st.info(f'This is a transfer evaluation: the {chosen_model} model was trained for another depot. Compatibility does not guarantee good accuracy for {name}. Compare errors against a baseline before deciding to use it.')
                else:
                    st.error('NOT READY — Fix the failed checks above or select another saved model. Evaluation is unavailable until all mandatory checks pass.')
                selected_metrics = _selected_metric_columns()
                if st.button(f'Evaluate {chosen_model} model on uploaded data',disabled=not compatible,key='evaluate_uploaded_model'):
                    with st.spinner(f'Evaluating {chosen_model} saved LSTM on continuous 96-reading input windows'):
                        result, baseline, matched = _evaluate_with_baseline(chosen_model,incoming,capacity)
                    st.session_state[f'eval_{_slug(name)}']={'hash':hashlib.sha256(blob).hexdigest(),'model':chosen_model,'scores':result,'baseline':baseline,'matched':matched}
                evaluated=st.session_state.get(f'eval_{_slug(name)}')
                if evaluated and evaluated.get('hash')==hashlib.sha256(blob).hexdigest() and evaluated.get('model')==chosen_model:
                    st.markdown(f'**Evaluation results — {chosen_model} saved LSTM**')
                    st.dataframe(pd.DataFrame([{k:evaluated['scores'].get(k) for k in selected_metrics} | {'Observations':evaluated['scores'].get('Observations')}]),hide_index=True,use_container_width=True)
                    if evaluated.get('baseline') and evaluated.get('matched'):
                        st.markdown('**Matched previous-day baseline comparison**')
                        st.dataframe(pd.DataFrame([{'Model':'Saved LSTM', **{k:evaluated['matched'].get(k) for k in selected_metrics}},
                                                   {'Model':'Previous-day baseline', **{k:evaluated['baseline'].get(k) for k in selected_metrics}}]),
                                     hide_index=True,use_container_width=True)
                    verdict, interpretation = _uploaded_model_decision(evaluated.get('matched') or evaluated['scores'], evaluated.get('baseline'))
                    with st.container(border=True):
                        st.markdown('#### Model performance interpretation')
                        st.markdown(f'**{verdict}**')
                        st.write(interpretation)
                        st.caption('Advisory only: these criteria do not certify model safety or multi-day forecasting performance.')
                    st.caption('Technical compatibility is not model approval. Evaluate accuracy against a previous-day baseline and assess multi-day forecasting separately before activation.')
                st.markdown('#### Step 4 — Train a new model using the uploaded observations')
                st.caption('Choose this option to train a separate model from the uploaded/merged data. Your active model stays unchanged until you explicitly activate the new version.')
                minimum_rows = st.slider('Minimum eligible training observations', min_value=1000, max_value=200000,
                                         value=1000, step=1000, key='optivolt_minimum_training_rows',
                                         help='Applies to eligible training targets after the 70/15/15 split and 96-reading windows.')
                readiness=training_readiness(merged, minimum_rows)
                st.markdown('**Training dataset eligibility**')
                with st.container(key='ov_training_readiness_metrics'):
                    r1,r2,r3,r4=st.columns(4)
                    r1.metric('Valid 15-minute rows',f"{readiness['valid_rows']:,}")
                    r2.metric('Eligible training rows',f"{readiness['train_rows']:,} / {minimum_rows:,}")
                    r3.metric('Validation targets',f"{readiness['validation_rows']:,}")
                    r4.metric('Held-out test targets',f"{readiness['test_rows']:,}")
                st.progress(min(readiness['train_rows']/minimum_rows,1.0),
                            text=f"Training observations: {readiness['train_rows']:,} of {minimum_rows:,} minimum")
                st.caption('Chronological split: 70% training, 15% validation, 15% held-out test. '
                           'The 96-reading LSTM input windows remain unchanged. '
                           f'At least {minimum_rows:,} eligible training targets are required. '
                           'Uploading data never retrains the saved model.')
                if readiness['ready']:
                    st.success('PASS — Enough eligible training observations to train a new LSTM.')
                else:
                    st.warning(f"NOT READY — {max(0,minimum_rows-readiness['train_rows']):,} more eligible training observations needed. "
                               'Upload additional continuous 15-minute readings. Validation and test each require at least 96 targets.')
                    if readiness['detail']: st.caption(readiness['detail'])
                epochs=st.number_input('Maximum training epochs',min_value=1,max_value=60,value=8,step=1)
                confirm=st.checkbox('I want to train a new model and preserve all existing versions')
                if st.button('Train new LSTM model',disabled=(not confirm or not readiness['ready']),type='primary'):
                    if not readiness['ready']:
                        st.error(f'At least {minimum_rows:,} eligible training observations are required.')
                    else:
                        status=st.empty()
                        bar=st.progress(0,text='Stage 1/6 · Preparing training data')
                        with st.spinner('Training and evaluating the new LSTM'):
                            result=train_candidate(name,merged,capacity,int(epochs),bar,status=status,minimum_training_rows=minimum_rows)
                        save_data(name,merged,capacity)
                        st.session_state[f'candidate_{_slug(name)}']=result
                        st.success(f"New model {result['version'].upper()} saved successfully. It is NOT active yet.")
                new_model=st.session_state.get(f'candidate_{_slug(name)}')
                if new_model:
                    st.markdown(f"#### New model {new_model['version'].upper()} — held-out evaluation")
                    rows=[]
                    if new_model.get('previous_active_metrics'):
                        rows.append({'Model':'Existing active model',**new_model['previous_active_metrics']})
                    rows.append({'Model':f"New model {new_model['version'].upper()}",**new_model['candidate_metrics']})
                    if new_model.get('previous_day_baseline_metrics'):
                        rows.append({'Model':'Previous-day baseline',**new_model['previous_day_baseline_metrics']})
                    st.dataframe(pd.DataFrame(rows)[['Model'] + [k for k in selected_metrics if k in pd.DataFrame(rows).columns] + ['Observations']],hide_index=True,use_container_width=True)
                    st.caption('MAE and RMSE: lower is better. R²: closer to 1 is better. Breach recall: higher is better. False alarm ratio: lower is better. Metrics are calculated on held-out observations.')
                    verdict,explanation=_model_decision(new_model['candidate_metrics'],new_model.get('previous_active_metrics'),new_model.get('previous_day_baseline_metrics'))
                    if verdict.startswith('Keep'):st.warning('**'+verdict+'** · '+explanation)
                    else:st.info('**'+verdict+'** · '+explanation)
                    st.markdown('**Next step:** Review the comparison, then choose a model version below. Activating a model is always a separate action.')
                    if st.button('Demand Prediction →', type='primary', key=f'prediction_after_training_{_slug(name)}'):
                        destination=st.session_state.get('_optivolt_pages_by_section',{}).get('📈 Demand Forecast')
                        if destination is not None:
                            st.switch_page(destination)
                        else:
                            st.info('Demand Prediction is available from the sidebar after activating a saved model.')
            except Exception as exc:
                st.error(f'Data operation failed: {exc}')
    with st.container(key="ov_depot_card_models"):
        st.markdown('#### Model version control')
        versions=available_versions(name)
        active=info.get('active','original')
        st.write(f'Active model: **{_version_label(active)}**')
        if versions:
            choice=st.selectbox('Choose model version',versions,format_func=_version_label,key=f'activate_version_{_slug(name)}')
            if st.button('Activate selected version',disabled=choice==active):
                activate(name,choice)
                st.success(f'Activated {choice}. Previous models remain saved.')
                st.rerun()
            if st.button('Keep current active version'):
                st.info('No changes made to the active model.')
            # Building a ZIP on every widget interaction made the upload page slow.
            # Prepare it only when requested; the user can then download it.
            backup_key=f'prepared_backup_{_slug(name)}'
            version_key=f'{choice}|{model_revision(name)}|{dataset_revision(name)}'
            prepared=st.session_state.get(backup_key)
            if st.button('Prepare model and data backup', key=f'prepare_backup_{_slug(name)}'):
                with st.spinner('Preparing downloadable model and observations'):
                    prepared={'version':version_key,'bytes':version_export(name,choice)}
                    st.session_state[backup_key]=prepared
            if prepared is not None and prepared.get('version')==version_key:
                st.download_button('Download selected model and data for permanent backup',
                                   prepared['bytes'],file_name=f'{_slug(name)}_{choice}_backup.zip',
                                   mime='application/zip',key=f'download_backup_{_slug(name)}')
        else:
            st.info('New depot has no saved model yet. Upload data and train a new model before activation.')

@st.cache_resource(show_spinner=False,max_entries=16)
def _cached_active_model(name, revision):
    # Compile the inference graph only once for each saved-model revision.
    # This avoids TensorFlow tracing on every horizon change and rerun.
    import tensorflow as tf
    model, scalers = _load_model(name)
    @tf.function(input_signature=[tf.TensorSpec((1,96,1),tf.float32)])
    def inference_step(x):
        return model(x, training=False)
    return inference_step, scalers

@st.cache_data(show_spinner=False,max_entries=32)
def _cached_forecast(name,model_rev,data_rev,hours):
    # Genuine recursive inference, cached against current active model + data versions.
    import tensorflow as tf
    inference_step,(xm,xs,ym,ys)=_cached_active_model(name,model_rev)
    df=observations(name)
    history=_regular(df)
    window=history.iloc[-96:].to_numpy(np.float32).copy()
    if len(window)!=96:raise ValueError('At least 96 observations required')
    if (history.index[-1]-history.index[-96]) != pd.Timedelta(minutes=95*15):
        raise ValueError('The latest 24-hour input window contains a gap. Upload consecutive 15-minute records.')
    num=int(hours*4); forecast=np.empty(num,float)
    for i in range(num):
        scaled=((window-xm)/xs).reshape(1,96,1)
        value=float(inference_step(tf.convert_to_tensor(scaled,dtype=tf.float32)).numpy().reshape(-1)[0])*ys+ym
        value=max(value,0.)
        forecast[i]=value
        window[:-1]=window[1:];window[-1]=value
    dates=pd.date_range(history.index.max()+pd.Timedelta(minutes=15),periods=num,freq='15min')
    return history.tail(96*2),pd.Series(forecast,index=dates)

@st.cache_data(show_spinner=False,max_entries=48)
def _cached_synthetic_forecast(name, model_rev, data_rev, start_date, growth_pct, hours):
    """Run a real saved LSTM on a clearly labelled synthetic 24-hour demand seed."""
    import tensorflow as tf
    from future_simulation import synthetic_seed

    seed = synthetic_seed(observations(name), start_date, float(growth_pct))
    inference_step, (xm, xs, ym, ys) = _cached_active_model(name, model_rev)
    window = seed.to_numpy(dtype=np.float32).copy()
    forecast = np.empty(int(hours * 4), dtype=float)
    for i in range(len(forecast)):
        scaled = ((window - xm) / xs).reshape(1, 96, 1)
        result = inference_step(tf.convert_to_tensor(scaled, dtype=tf.float32))
        value = max(0.0, float(result.numpy().reshape(-1)[0]) * ys + ym)
        forecast[i] = value
        window[:-1] = window[1:]
        window[-1] = value
    times = pd.date_range(pd.Timestamp(start_date), periods=len(forecast), freq='15min')
    return seed, pd.Series(forecast, index=times, name='LSTM simulated-scenario forecast (kW)')




def _sixty_day_profile(name, target_times, growth_pct=0):
    """Historical weekly quarter-hour profile, using up to the latest 60 full days.

    This is a statistical planning profile, not an LSTM trained with 5,760 inputs.
    """
    historical = _regular(observations(name)).sort_index()
    last = historical.index[-1]
    cutoff = last - pd.Timedelta(days=60)
    recent = historical.loc[historical.index > cutoff].dropna()
    if recent.empty or (recent.index[-1] - recent.index[0]) < pd.Timedelta(days=59):
        raise ValueError('A 30-day planning forecast needs approximately 60 days of historical 15-minute readings. Upload more depot data.')
    # A weekday/quarter-hour profile retains weekly charging patterns.
    keys = pd.MultiIndex.from_arrays(
        [recent.index.dayofweek, recent.index.hour * 4 + recent.index.minute // 15]
    )
    medians = pd.Series(recent.to_numpy(dtype=float), index=keys).groupby(level=[0, 1]).median()
    fallback = recent.groupby(recent.index.hour * 4 + recent.index.minute // 15).median()
    overall = float(recent.median())
    values = np.array([
        float(medians.get((t.dayofweek, t.hour * 4 + t.minute // 15),
                          fallback.get(t.hour * 4 + t.minute // 15, overall)))
        for t in target_times
    ], dtype=float)
    return np.maximum(0.0, values * (1.0 + float(growth_pct) / 100.0))


@st.cache_data(show_spinner=False, max_entries=24)
def _cached_extended_forecast(name, model_rev, data_rev, hours, start_date=None, growth_pct=0):
    """Saved 96-input LSTM for 72h, then 60-day historical profile for planning."""
    short_hours = min(int(hours), 72)
    if start_date is None:
        hist, short = _cached_forecast(name, model_rev, data_rev, short_hours)
    else:
        hist, short = _cached_synthetic_forecast(
            name, model_rev, data_rev, str(start_date), int(growth_pct), short_hours
        )
    remaining = int(hours * 4) - len(short)
    if remaining <= 0:
        return hist, short
    future_index = pd.date_range(short.index[-1] + pd.Timedelta(minutes=15),
                                 periods=remaining, freq='15min')
    profile = _sixty_day_profile(name, future_index, growth_pct if start_date is not None else 0)
    extended = pd.Series(profile, index=future_index, name='60-day historical demand profile (kW)')
    return hist, pd.concat([short, extended])


def render_fast_forecast(name):
    """Fast LSTM forecast with transparent operational and future scenario modes.

    Future scenarios use a labelled synthetic 24-hour input created from historical profiles.
    """
    import plotly.graph_objects as go
    from datetime import date, timedelta
    mode = st.radio('Future forecasting method',
                    ['Historical demand · saved LSTM', 'Fleet & charging-aware · train / validate / predict'],
                    horizontal=True, key=f'future_method_{name}')
    if mode.startswith('Fleet'):
        try:
            import growth_aware_workflow
            growth_aware_workflow.render(name, observations(name))
        except Exception as exc:
            st.error(f'Fleet-aware forecasting unavailable: {exc}')
        return

    st.markdown('''<style>
    /* Compact, high-contrast styling for the demand forecasting page. */
    div[data-testid="stAppViewContainer"] {background: linear-gradient(160deg,#F1F6FF 0%,#F8FAFF 55%,#EDF9F7 100%);}
    div[data-testid="stAppViewContainer"] h1 {color:#12366B !important; letter-spacing:-.03em;}
    div[data-testid="stAppViewContainer"] h2,div[data-testid="stAppViewContainer"] h3 {color:#164579 !important;}
    div[data-testid="stAppViewContainer"] label,div[data-testid="stAppViewContainer"] p,
    div[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {color:#253B57;}
    div[data-testid="stVerticalBlock"] {gap:.65rem;}
    div[data-testid="stElementContainer"] {margin-bottom:0 !important;}
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] {background:linear-gradient(135deg,#E6F1FF,#EFFBFC);border:1px solid #B9D5F6;border-top:4px solid #2563EB;border-radius:14px;padding:14px 16px;}
    div[data-testid="stMetricValue"] {color:#102E54 !important;}
    div[data-testid="stMetricLabel"] {color:#25466D !important;}
    div[data-testid="stAppViewContainer"] button[kind="primary"] {background:#1857B8;color:white;border-radius:10px;}
    div[data-testid="stDataFrame"] {border:1px solid #B9D5F6;border-radius:10px;}
    .optivolt-summary {background:linear-gradient(100deg,#DDEBFF,#DDF5F1);color:#12385C;
      padding:13px 17px;border-radius:12px;border-left:5px solid #1679BD;font-size:15px;line-height:1.45;}
    /* A denser dashboard rhythm: avoid oversized empty bands. */
    div[data-testid="stMainBlockContainer"] {padding-top:1.15rem!important;padding-bottom:1.5rem!important;}
    div[data-testid="stMainBlockContainer"] > div {gap:.45rem!important;}
    div[data-testid="stVerticalBlock"] {gap:.42rem!important;}
    div[data-testid="stHorizontalBlock"] {gap:.8rem!important;}
    div[data-testid="stWidgetLabel"] {margin-bottom:.15rem!important;}
    div[data-testid="stRadio"] [role="radiogroup"] {gap:.35rem!important;}
    div[data-testid="stExpander"] {border:1px solid #bfd6ef;border-radius:12px;background:#f6faff;}
    div[data-testid="stAppViewContainer"] h1 {font-size:2rem!important;margin:.15rem 0 .35rem!important;}
    div[data-testid="stAppViewContainer"] h2,div[data-testid="stAppViewContainer"] h3 {font-size:1.22rem!important;margin:.45rem 0 .2rem!important;}
    div[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {font-size:.85rem!important;}
    div[data-testid="stAppViewContainer"] [data-testid="stDataFrame"] {box-shadow:0 5px 16px rgba(16,52,91,.07);overflow:hidden;}
    div[data-testid="stAppViewContainer"] [data-testid="stDataFrame"] * {color:#102a4a!important;}
    div[data-testid="stAppViewContainer"] [data-testid="stMetric"] {padding:10px 14px!important;}
    /* Strong white-on-blue section titles: override the page-wide heading rules. */
    div[data-testid="stAppViewContainer"] .ov-hero h1,
    div[data-testid="stAppViewContainer"] .opt-diag-hero h3 {color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;text-shadow:0 1px 2px rgba(0,0,0,.15);}
    div[data-testid="stAppViewContainer"] .ov-section,
    div[data-testid="stAppViewContainer"] .ov-section strong,
    div[data-testid="stAppViewContainer"] .ov-section small {color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;}
    div[data-testid="stAppViewContainer"] .opt-diag-hero p {color:#F0FBFF!important;-webkit-text-fill-color:#F0FBFF!important;}
    /* Stronger header contrast for Streamlit's dataframe grid. */
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="columnheader"] *,
    div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {color:#102A4A!important;font-weight:800!important;}
    div[data-testid="stDataFrame"] [role="gridcell"],
    div[data-testid="stDataFrame"] [role="gridcell"] * {color:#102A4A!important;font-weight:650!important;}
    .ov-hero {background:linear-gradient(110deg,#12366B,#126A91 67%,#078D92);color:#fff;
      border-radius:16px;padding:17px 23px;margin:0 0 9px;box-shadow:0 8px 22px rgba(14,53,100,.16);}
    .ov-hero h1 {font-size:1.85rem!important;color:#fff!important;margin:0 0 4px!important;line-height:1.1!important;}
    .ov-hero p {color:#eaf7ff!important;margin:0!important;font-size:.94rem;}
    .ov-section {background:linear-gradient(105deg,#163e77,#0c8290);color:#fff;
      padding:11px 16px;border-radius:12px;margin:9px 0 5px;display:flex;align-items:center;gap:10px;
      box-shadow:0 5px 15px rgba(18,61,109,.13);}
    .ov-section strong {font-size:1.08rem;color:#fff!important;letter-spacing:.005em;}
    .ov-section small {font-size:.83rem;color:#e6f8ff!important;margin-left:auto;}
    .ov-pill {display:inline-block;background:#dff5f3;border:1px solid #8acbc6;border-radius:20px;
      padding:3px 10px;color:#11565c;font-weight:700;font-size:.8rem;}
    .ov-period {background:#e9f2ff;border-left:4px solid #2465c8;border-radius:10px;
      padding:9px 13px;color:#17365d;margin:4px 0;font-size:.92rem;}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="ov-hero"><h1>⚡ Demand Forecast</h1><p>Saved LSTM · 15-minute predictions · Forecast and fleet-growth planning</p></div>', unsafe_allow_html=True)
    v = registry().get(name, {})
    st.markdown(f'<span class="ov-pill">Depot: {name} · Model: {v.get("active", "original")}</span>', unsafe_allow_html=True)

    mode = st.radio(
        'Forecast purpose',
        ['Future demand prediction', 'Forecast after latest observations'],
        key=f'forecast_purpose_{_slug(name)}',
        help='A future-date scenario uses historical demand as its input pattern, not measurements from the chosen future date.',
    )
    a, b = st.columns(2)
    with a:
        hours = st.selectbox('Forecast horizon', [2, 4, 6, 12, 24, 48, 72, 168, 336, 720],
                             index=4, format_func=lambda h: f'{h // 24} days' if h >= 168 else f'{h} hours',
                             key=f'fast_horizon_{_slug(name)}')
    with b:
        capacity = st.number_input('Grid capacity (kW)', min_value=.1,
                                   value=float(v.get('capacity_kw', BUILTIN.get(name, 17.))),
                                   step=1., key=f'fast_grid_{_slug(name)}')

    import long_term_lstm
    long_term_ready = long_term_lstm.model_available(name)
    method = st.selectbox(
        'Forecasting method',
        ['Existing LSTM + historical profile'] +
        (['Dedicated 30-day LSTM (trained)'] if long_term_ready else []),
        help='The dedicated model appears only after its trained model.keras and metadata.json are present.',
        key=f'long_term_method_{_slug(name)}',
    )
    if not long_term_ready:
        st.caption('Dedicated 30-day LSTM not trained for this depot yet. The existing hybrid method remains available.')
    # Explicitly distinguish the two trained LSTM approaches for demonstrations.
    st.markdown('### Two LSTM forecasting approaches')
    st.markdown('**LSTM 1 — Short-term:** 24-hour (96-reading) input; recursive forecasts up to 72 hours.  '
                '**LSTM 2 — Long-term:** 60-day input; direct multi-output forecasts up to 30 days. '
                'Longer forecasts using LSTM 1 are **hybrid**, not a third trained LSTM.')
    st.caption('The five evaluation criteria below refer to held-out observed demand.')
    try:
        import json as _json
        from pathlib import Path as _Path
        _meta_file = (_Path(__file__).resolve().parent / 'models' / 'long_term_lstm' /
                      name.replace(' ', '_').lower() / 'metadata.json')
        _long_meta = _json.loads(_meta_file.read_text()) if _meta_file.is_file() else {}
    except (OSError, ValueError, TypeError):
        _long_meta = {}
    def _show_metric(value, unit='', digits=3):
        try:
            return f'{float(value):,.{digits}f}{unit}' if value is not None else 'Not yet evaluated'
        except (TypeError, ValueError):
            return 'Not yet evaluated'
    _long_metrics = {
        'MAE (kW)': _show_metric(_long_meta.get('test_mae_kw'), ' kW'),
        'RMSE (kW)': _show_metric(_long_meta.get('test_rmse_kw'), ' kW'),
        'R²': _show_metric(_long_meta.get('test_r2')),
        'MAPE (%)': _show_metric(_long_meta.get('test_mape_pct'), '%'),
        'Peak demand error (kW)': _show_metric(_long_meta.get('test_peak_error_kw'), ' kW'),
    }
    # The 72-hour evaluator writes separate metadata; trained models stay untouched.
    try:
        _short_meta_file = (_Path(__file__).resolve().parent / 'models' /
                            'short_term_lstm_evaluation' /
                            (name.lower().replace(' ', '_').replace('-', '_') + '.json'))
        _short_meta = _json.loads(_short_meta_file.read_text(encoding='utf-8')) if _short_meta_file.is_file() else {}
    except (OSError, ValueError, TypeError):
        _short_meta = {}
    _short_metrics = {
        'MAE (kW)': _show_metric(_short_meta.get('test_mae_kw'), ' kW'),
        'RMSE (kW)': _show_metric(_short_meta.get('test_rmse_kw'), ' kW'),
        'R²': _show_metric(_short_meta.get('test_r2')),
        'MAPE (%)': _show_metric(_short_meta.get('test_mape_pct'), '%'),
        'Peak demand error (kW)': _show_metric(_short_meta.get('test_peak_error_kw'), ' kW'),
    }
    # Side-by-side, read-only scorecards. Values come from the saved evaluation JSON.
    # This is a display-only change: model selection and forecasting remain untouched.
    import html as _html
    def _metric_card(title, subtitle, values):
        def _cell(label, value):
            return (
                '<div style="background:#21466f;border:1px solid #355b84;'
                'border-radius:10px;padding:10px 12px;min-width:0;">'
                '<div style="font-size:12px;color:#d5e6fa;">' + _html.escape(label) + '</div>'
                '<div style="font-size:19px;font-weight:750;color:#fff;overflow-wrap:anywhere;">'
                + _html.escape(value) + '</div></div>'
            )
        main = _cell('MAE · average error', values['MAE (kW)'])
        extras = ''.join(_cell(label, values[label]) for label in (
            'RMSE (kW)', 'R²', 'MAPE (%)', 'Peak demand error (kW)'))
        return (
            '<div style="background:#17365d;border:1px solid #3478f6;'
            'border-top:4px solid #3478f6;border-radius:15px;padding:16px;'
            'color:#fff;min-height:100%;">'
            '<div style="font-size:19px;font-weight:750;color:#fff;">'
            + _html.escape(title) + '</div>'
            '<div style="font-size:13px;color:#d5e6fa;margin:3px 0 14px;">'
            + _html.escape(subtitle) + '</div>'
            + main + '<div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));'
            'gap:8px;margin-top:8px;">' + extras + '</div></div>'
        )
    _left_card, _right_card = st.columns(2, gap='medium')
    with _left_card:
        st.markdown(_metric_card('LSTM 1 · Short-term', 'Recursive forecast · 72 hours', _short_metrics),
                    unsafe_allow_html=True)
    with _right_card:
        st.markdown(_metric_card('LSTM 2 · Long-term', 'Direct forecast · 30 days', _long_metrics),
                    unsafe_allow_html=True)
    st.caption('**Why the evaluation matters:** These scores come from held-out UKPN demand. '
               'Lower forecast and peak errors provide stronger evidence of accuracy, '
               'but no score guarantees reliable future predictions.')
    with st.expander('What do these results tell us about model confidence?', expanded=False):
        st.markdown(
            '- **MAE:** Average absolute error in kW. Lower is better.\n'
            '- **RMSE:** Penalises large mistakes, particularly near the grid limit.\n'
            '- **R²:** Measures performance against predicting average demand; negative values are a warning.\n'
            '- **MAPE:** Percentage error; low demand can exaggerate the result.\n'
            '- **Peak demand error:** Measures how accurately peaks are forecast, '
            'which matters for grid-capacity warnings.'
        )
        st.markdown('**Confidence assessment:** Evaluate on unseen real observations, '
                    'check for missed demand peaks and compare against a simple baseline. '
                    'A model should not be labelled highly reliable on these metrics alone.')
    with st.expander('How the five metrics are calculated', expanded=False):
        st.markdown('**MAE** is mean absolute error; **RMSE** is root mean squared error; '
                    '**R²** compares errors with the observed mean; **MAPE** averages '
                    'absolute percentage errors where actual demand exceeds 0.1 kW; '
                    '**peak demand error** averages absolute differences in forecast-window peaks.')
        st.caption('LSTM 1 is evaluated using recursive 72-hour forecasts; '
                   'LSTM 2 is evaluated using direct 30-day forecasts. '
                   'Run evaluate_short_term_lstm.py --all to populate LSTM 1 results.')
    scenario = mode.startswith('Future demand prediction')
    future_date = None
    if scenario:
        date_col, growth_col = st.columns([1, 1.5], gap='medium')
        with date_col:
            future_date = st.date_input(
                'Prediction start date',
                value=date.today() + timedelta(days=1),
                min_value=date.today(),
                key=f'future_prediction_start_{_slug(name)}',
            )
        with growth_col:
            growth_pct = st.slider('Fleet-demand growth (%)', min_value=0, max_value=150,
                                   value=20, step=5, key=f'future_growth_{_slug(name)}')
        st.markdown('<div class="optivolt-summary"><strong>Future demand prediction</strong> · '
                    'Saved LSTM (first 72 hours) · 60-day historical profile for longer forecasts</div>',
                    unsafe_allow_html=True)
        with st.expander('How the future prediction is generated', expanded=False):
            st.write('The saved LSTM uses 96 consecutive 15-minute readings (24 hours) as its fixed input. '
                     'The forecast horizon is selected separately (up to 30 days). The first 72 hours '
                     'use recursive LSTM predictions; the remaining period uses the previous 60 days '
                     'of historical weekday and time-of-day demand patterns.')
            st.write('Future inputs are derived from historical demand on matching weekdays and adjusted '
                     'for the selected growth assumption. The historical-profile extension is a planning '
                     'projection, not a separately trained LSTM.')

    if method.startswith('Dedicated') and scenario:
        st.warning('The dedicated 30-day LSTM currently supports forecasts immediately after the latest complete observed day only. Choose "Forecast after latest observations" to use it.')
        return

    key = f'run_fast_forecast_{_slug(name)}'
    if not st.session_state.get(key):
        st.info('Choose the forecast settings and click Generate forecast. '
                'The saved LSTM is loaded only when you request a prediction.')
        if st.button('Generate forecast', type='primary', key=f'generate_{_slug(name)}'):
            st.session_state[key] = True
            st.rerun()
        return

    try:
        with st.spinner('Running saved LSTM inference for the selected horizon'):
            if method.startswith('Dedicated'):
                hist, pred = long_term_lstm.forecast(name, _regular(observations(name)), int(hours))
            elif scenario:
                seed, pred = _cached_extended_forecast(
                    name, model_revision(name), dataset_revision(name), int(hours),
                    str(future_date), int(growth_pct))
                hist = seed
            else:
                hist, pred = _cached_extended_forecast(
                    name, model_revision(name), dataset_revision(name), int(hours))

        if hours > 72 and not method.startswith('Dedicated'):
            st.info('Forecast method: first 72 hours use the saved 96-input LSTM; the remaining period uses a 60-day historical weekday/time-of-day demand profile. The saved model is not retrained.')
        if scenario:
            st.markdown(f'<div class="ov-period"><strong>Prediction period · {hours} hours</strong> &nbsp; {pred.index[0]:%d %b %Y %H:%M} – {(pred.index[-1] + pd.Timedelta(minutes=15)):%d %b %Y %H:%M} (end exclusive)</div>', unsafe_allow_html=True)
            st.caption('Simulation input: 96 synthetic 15-minute readings for the day immediately before '
                       'the selected future date, derived from historical complete days on the matching '
                       f'weekday (reference: {seed.attrs.get("historical_reference_day", "recorded data")}) '
                       f'with a {growth_pct:+d}% demand adjustment. The saved LSTM is not retrained.')
        else:
            st.caption(f'Latest observed demand: {hist.index[-1]:%d %b %Y %H:%M}. '
                       'Predictions begin at the next 15-minute interval.')
            if hist.index[-1] < pd.Timestamp.now() - pd.Timedelta(days=7):
                st.warning('These depot observations are not recent. This is a historical continuation, '
                           'not a current operational forecast. Upload recent readings to forecast from today.')

        import forecast_results_store
        try:
            forecast_results_store.save(name, pred, capacity, 'Saved LSTM · future demand')
        except Exception as exc:
            st.warning(f'Forecast is available here, but could not be saved for other pages: {exc}')

        peak = float(pred.max())
        breaches = pred >= capacity
        warnings = pred >= capacity * 0.8
        energy_kwh = float(pred.sum() * 0.25)
        peak_time = pred.idxmax()
        excess_kw = max(0.0, peak - capacity)
        breach_hours = float(breaches.sum()) * 0.25
        warning_hours = float(warnings.sum()) * 0.25
        a, b, c = st.columns(3)
        a.metric('Highest expected electricity demand', f'{peak:.2f} kW')
        b.metric('Maximum electricity available', f'{capacity:.1f} kW')
        c.metric('Maximum predicted capacity exceedance' if excess_kw > 0 else 'Minimum spare grid capacity',
                 f'{excess_kw:.2f} kW' if excess_kw > 0 else f'{capacity - peak:.2f} kW')
        d, e, f = st.columns(3)
        d.metric('Total electricity expected to be used', f'{energy_kwh:,.1f} kWh')
        e.metric('Time demand may reach or exceed the limit', f'{breach_hours:g} hours')
        f.metric('Busiest predicted time', peak_time.strftime('%d %b %H:%M'))

        # Plain-language interpretation. The forecast is a prediction, not a measured overload.
        # Inline colours keep the message readable even with the app's global Streamlit theme.
        if breaches.any():
            percent_above = (excess_kw / capacity * 100.0) if capacity > 0 else 0.0
            whole_window = bool(breaches.all())
            duration_text = (f'throughout the next {len(pred) * 0.25:g} hours'
                             if whole_window else f'for about {breach_hours:g} hours during this forecast')
            st.markdown(f"""
<div style="background:#112C52;border:2px solid #F2BA59;border-radius:16px;
            padding:20px 22px;color:#FFFFFF;-webkit-text-fill-color:#FFFFFF;margin:12px 0;">
  <div style="font-size:1.18rem;font-weight:750;color:#FFFFFF;-webkit-text-fill-color:#FFFFFF;">
    ⚠️ Warning: Predicted electricity demand exceeds grid capacity
  </div>
  <p style="color:#FFFFFF;-webkit-text-fill-color:#FFFFFF;margin:12px 0;">
    Your depot may need more electricity than its grid connection can provide {duration_text}.
    At the busiest point, demand could reach <b>{peak:.2f} kW</b>, but the available
    capacity is <b>{capacity:.2f} kW</b>. That is <b>{excess_kw:.2f} kW</b>
    ({percent_above:.0f}%) above the limit.
  </p>
  <div style="font-weight:750;color:#FFFFFF;-webkit-text-fill-color:#FFFFFF;">Recommended actions</div>
  <ul style="color:#FFFFFF;-webkit-text-fill-color:#FFFFFF;margin:8px 0 12px 0;">
    <li>Stagger vehicle charging so fewer vehicles charge at once.</li>
    <li>Reduce charging during the busiest periods using managed charging.</li>
    <li>Move flexible charging to quieter hours where possible.</li>
    <li>Check that the configured {capacity:.2f} kW grid limit is correct.</li>
    <li>If the problem continues, consider more grid capacity or on-site energy storage.</li>
  </ul>
  <p style="font-size:0.88rem;color:#FFFFFF;-webkit-text-fill-color:#FFFFFF;margin:0;">
    This is a forecast, not a confirmed electricity overload. Actual demand may differ.
  </p>
</div>""", unsafe_allow_html=True)
        elif warnings.any():
            st.info(f'Your depot is expected to stay within its {capacity:.2f} kW grid limit, '
                    f'but demand may reach at least 80% of that limit for about {warning_hours:g} hours. '
                    'Consider staggering charging during busy periods. '
                    'This is a forecast, not a measurement.')
        else:
            st.success(f'Your depot is expected to stay within its {capacity:.2f} kW grid limit '
                       'throughout this forecast. No high-demand warning is predicted. '
                       'Actual demand may differ.')

        fig = go.Figure()
        if scenario:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist.values, name='Scenario input (historical-pattern based)',
                mode='lines', line=dict(color='#2563EB', width=3),
                hovertemplate='%{x|%d %b %Y %H:%M}<br>Input: %{y:.2f} kW<extra></extra>'))
        else:
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist.values, name='Observed demand',
                line=dict(color='#1D4ED8', width=3)))
        fig.add_trace(go.Scatter(
            x=pred.index, y=pred.values,
            name='LSTM + historical-profile forecast' if hours > 72 else ('LSTM future scenario prediction' if scenario else 'LSTM forecast'),
            mode='lines', line=dict(color='#087F8C', width=4),
            fill='tozeroy', fillcolor='rgba(8,127,140,0.12)',
            hovertemplate='%{x|%d %b %Y %H:%M}<br>Prediction: %{y:.2f} kW<extra></extra>'))
        fig.add_trace(go.Scatter(
            x=[hist.index[0], pred.index[-1]], y=[capacity, capacity],
            name='Grid capacity', mode='lines', line=dict(color='#B91C1C', width=2.5, dash='dash'),
            hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=[hist.index[0], pred.index[-1]], y=[capacity * 0.8, capacity * 0.8],
            name='80% capacity warning', mode='lines', line=dict(color='#92400E', width=2.5, dash='dot'),
            hoverinfo='skip'))
        fig.update_layout(
            title=dict(text='Future demand prediction' if scenario else 'LSTM forecast after last observed interval',
                       font=dict(size=20, color='#0B1F3A')),
            height=520, font=dict(color='#111827', size=16),
            paper_bgcolor='#FFFFFF', plot_bgcolor='#F8FAFC',
            hovermode='x unified',
            xaxis=dict(tickfont=dict(color='#111827', size=15),
                       gridcolor='#CBD5E1', linecolor='#475569', showline=True,
                       title='Date and time', title_font=dict(color='#111827', size=16)),
            yaxis=dict(title='Demand (kW)', tickfont=dict(color='#111827', size=15),
                       gridcolor='#CBD5E1', linecolor='#475569', showline=True,
                       title_font=dict(color='#111827', size=16), rangemode='tozero'),
            legend=dict(orientation='h', y=1.13, x=0, font=dict(color='#111827', size=14),
                        bgcolor='rgba(255,255,255,0.95)'),
            margin=dict(l=55, r=30, b=60, t=105),
        )
        st.plotly_chart(fig, use_container_width=True)
        if scenario:
            with st.expander('Input data provenance and downloads', expanded=False):
                st.caption('The seed is simulated, constructed from historical demand; these are not future observations.')
                st.line_chart(seed, y_label='Simulated input demand (kW)')
                seed_table = pd.DataFrame({
                    'timestamp': seed.index,
                    'demand_kw': seed.values,
                    'data_source': 'synthetic_historical_weekday_median',
                    'demand_growth_pct': growth_pct,
                    'historical_reference_day': seed.attrs.get('historical_reference_day', ''),
                })
                st.download_button('Download synthetic input CSV', seed_table.to_csv(index=False),
                                   file_name=f'{_slug(name)}_synthetic_input_{future_date}.csv',
                                   mime='text/csv', key=f'download_synthetic_{_slug(name)}')
            result_table = pd.DataFrame({
                'timestamp': pred.index, 'predicted_demand_kw': pred.values,
                'prediction_type': ['LSTM_on_synthetic_input' if i < 72 * 4 else '60_day_historical_profile' for i in range(len(pred))],
                'scenario_growth_pct': growth_pct,
                'grid_capacity_kw': capacity,
                'grid_headroom_kw': capacity - pred.values,
                'capacity_exceeded': pred.values >= capacity,
            })
            st.download_button('Download future LSTM predictions CSV', result_table.to_csv(index=False),
                               file_name=f'{_slug(name)}_lstm_scenario_{future_date}.csv',
                               mime='text/csv', key=f'download_sim_predictions_{_slug(name)}')
            st.divider()
            st.markdown('<div class="ov-section"><strong>📈 Fleet-growth scenario comparison</strong><small>0% · 20% · 50% · 100%</small></div>', unsafe_allow_html=True)
            st.caption('Compare 0%, 20%, 50% and 100% fleet-growth scenarios using the saved LSTM. Each scenario uses a separate historical-pattern input.')
            comparison_key = f'compare_growth_{_slug(name)}'
            if st.button('Compare growth scenarios', key=f'compare_button_{_slug(name)}'):
                st.session_state[comparison_key] = True
            if st.session_state.get(comparison_key):
                with st.spinner('Generating four LSTM growth-scenario forecasts'):
                    comparison = {}
                    for growth in (0, 20, 50, 100):
                        _, scenario_pred = _cached_extended_forecast(
                            name, model_revision(name), dataset_revision(name), int(hours),
                            str(future_date), growth)
                        comparison[growth] = scenario_pred
                comparison_fig = go.Figure()
                scenario_colors = {0: '#1D4ED8', 20: '#087F8C', 50: '#7C3AED', 100: '#B45309'}
                for growth, scenario_pred in comparison.items():
                    comparison_fig.add_trace(go.Scatter(
                        x=scenario_pred.index, y=scenario_pred.values, mode='lines',
                        name=f'{growth}% demand growth',
                        line=dict(color=scenario_colors[growth], width=3, dash={0:'solid',20:'dash',50:'dot',100:'dashdot'}[growth]),
                        hovertemplate='%{x|%d %b %Y %H:%M}<br>Demand: %{y:.3f} kW<extra>' + str(growth) + '% growth</extra>'))
                comparison_fig.add_hline(y=capacity, line_color='#B91C1C',
                                         line_dash='dash', annotation_text='Grid capacity',
                                         annotation_font_color='#7F1D1D')
                comparison_fig.add_hline(y=capacity * 0.8, line_color='#92400E',
                                         line_dash='dot', annotation_text='80% warning',
                                         annotation_font_color='#78350F')
                comparison_fig.update_layout(
                    title='Future demand prediction by fleet-growth scenario',
                    height=440, font=dict(size=15, color='#111827'),
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#F8FAFC',
                    xaxis=dict(title='Date and time', tickfont=dict(color='#111827', size=14),
                               gridcolor='#CBD5E1'),
                    yaxis=dict(title='Predicted demand (kW)', tickfont=dict(color='#111827', size=14),
                               gridcolor='#CBD5E1', rangemode='tozero'),
                    legend=dict(orientation='h', y=1.12, x=0, font=dict(size=14, color='#111827')),
                    margin=dict(l=65, r=35, t=100, b=65), hovermode='x unified')
                st.plotly_chart(comparison_fig, use_container_width=True)
                # Differences are plotted separately because the recursive model can
                # converge to nearly identical outputs for distinct growth assumptions.
                baseline = comparison[0]
                delta_fig = go.Figure()
                for growth in (20, 50, 100):
                    delta = comparison[growth].subtract(baseline)
                    delta_fig.add_trace(go.Scatter(
                        x=delta.index, y=delta.values, mode='lines',
                        name=f'{growth}% vs 0% growth',
                        line=dict(color=scenario_colors[growth], width=3),
                        hovertemplate='%{x|%d %b %Y %H:%M}<br>Difference: %{y:+.4f} kW<extra></extra>'))
                delta_fig.add_hline(y=0, line_color='#334155', line_width=1)
                delta_fig.update_layout(
                    title='Change in prediction relative to 0% growth', height=340,
                    font=dict(size=15, color='#0F172A'),
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#F8FAFC',
                    xaxis=dict(title='Date and time', tickfont=dict(color='#0F172A', size=14), gridcolor='#CBD5E1'),
                    yaxis=dict(title='Demand difference (kW)', tickfont=dict(color='#0F172A', size=14), gridcolor='#CBD5E1', zeroline=True),
                    legend=dict(orientation='h', y=1.15, x=0, font=dict(size=14, color='#0F172A')),
                    margin=dict(l=75, r=30, t=95, b=65), hovermode='x unified')
                st.plotly_chart(delta_fig, use_container_width=True)
                max_difference = max(float((comparison[g] - baseline).abs().max()) for g in (20, 50, 100))
                # Keep the numerical sensitivity result in the diagnostics; avoid a large warning banner here.
                summary = pd.DataFrame([
                    {'Demand growth (%)': growth,
                     'Peak demand (kW)': round(float(series.max()), 2),
                     'Total energy (kWh)': round(float(series.sum()) * 0.25, 1),
                     'Minimum headroom (kW)': round(float(capacity - series.max()), 2),
                     'Hours at/above capacity': round(float((series >= capacity).sum()) * 0.25, 2),
                     'Hours at/above 80%': round(float((series >= capacity * 0.8).sum()) * 0.25, 2)}
                    for growth, series in comparison.items()
                ])
                st.markdown('''<style>
                div[data-testid="stDataFrame"] {color: #0F172A !important;}
                div[data-testid="stDataFrame"] * {color: #0F172A !important; font-weight: 550;}
                </style>''', unsafe_allow_html=True)
                st.dataframe(summary.style.set_properties(**{'color':'#112B4A','background-color':'#E7F1FC','font-weight':'700'}).set_table_styles([{'selector':'th','props':[('color','#102D50'),('background-color','#BEDCF9'),('font-weight','800')]}]).format({
                    'Peak demand (kW)': '{:.2f}', 'Total energy (kWh)': '{:.1f}',
                    'Minimum headroom (kW)': '{:.2f}',
                    'Hours at/above capacity': '{:.2f}', 'Hours at/above 80%': '{:.2f}'
                }), hide_index=True, use_container_width=True)
                st.download_button('Download scenario comparison CSV', summary.to_csv(index=False),
                                   file_name=f'{_slug(name)}_growth_comparison_{future_date}.csv',
                                   mime='text/csv', key=f'download_growth_comparison_{_slug(name)}')
                st.caption('Scenario differences reflect the assumed demand-growth adjustment and recursive '
                           'LSTM response. They do not quantify predictive uncertainty.')
            st.markdown("""
            <style>
            .opt-diag-hero {background:linear-gradient(115deg,#112b62 0%,#125b8a 55%,#087f8c 100%);border-radius:14px;padding:13px 19px;color:#fff;margin:8px 0 9px;box-shadow:0 8px 20px rgba(12,46,90,.16)}
            .opt-diag-hero h3 {color:#fff!important;-webkit-text-fill-color:#fff!important;margin:0 0 7px!important;font-size:1.45rem!important}
            .opt-diag-hero p {color:#e9f8ff!important;margin:0!important;font-size:1rem;line-height:1.55}
            .opt-diag-card {background:#eef6ff;border:1px solid #c9e1fa;border-left:5px solid #1672b8;border-radius:12px;padding:10px 14px;margin:6px 0;color:#173455}
            .opt-diag-card.amber {background:#fff6e7;border-color:#f5d8a4;border-left-color:#c46b09}
            .opt-diag-card.green {background:#eafaf3;border-color:#b8e7d0;border-left-color:#16845d}
            .opt-diag-card strong {color:#102a4a;font-size:1.03rem}
            .opt-diag-card span {color:#29435e;line-height:1.55}
            div[data-testid="stAppViewContainer"] .opt-diag-hero h3 {color:#fff!important;-webkit-text-fill-color:#fff!important;}
            div[data-testid="stAppViewContainer"] .opt-diag-hero p {color:#f0fbff!important;-webkit-text-fill-color:#f0fbff!important;}
            </style>
            <div class="opt-diag-hero">
              <h3>🔬 Model Intelligence &amp; Forecast Reliability</h3>
              <p>Understand how fleet growth influences the LSTM, and whether its response fades as predictions are fed back into the model.</p>
            </div>
            """, unsafe_allow_html=True)
            with st.expander('What does this diagnostic measure?', expanded=False):
                st.markdown('The chart plots the **difference in predicted demand (kW)** between each growth scenario and the 0% baseline. A value near zero means the saved LSTM produces similar predictions under different growth inputs. This is a **responsiveness test**, not an accuracy score or a confidence interval. The model is not retrained.')
            if st.button('Run model diagnostics', key=f'diagnostics_button_{_slug(name)}'):
                st.session_state[f'diagnostics_run_{_slug(name)}'] = True
            if st.session_state.get(f'diagnostics_run_{_slug(name)}'):
                with st.spinner('Testing LSTM sensitivity across forecast horizons'):
                    diagnostic_predictions = {}
                    diagnostic_inputs = {}
                    for diagnostic_growth in (0, 20, 50, 100):
                        diagnostic_seed, diagnostic_pred = _cached_synthetic_forecast(
                            name, model_revision(name), dataset_revision(name),
                            str(future_date), diagnostic_growth, int(hours))
                        diagnostic_inputs[diagnostic_growth] = diagnostic_seed
                        diagnostic_predictions[diagnostic_growth] = diagnostic_pred
                base_seed = diagnostic_inputs[0]
                base_pred = diagnostic_predictions[0]
                records = []
                for diagnostic_growth in (0, 20, 50, 100):
                    seed_values = diagnostic_inputs[diagnostic_growth].to_numpy(dtype=float)
                    pred_values = diagnostic_predictions[diagnostic_growth].to_numpy(dtype=float)
                    baseline_values = base_pred.to_numpy(dtype=float)
                    for checkpoint_hours in (0.25, 1, 6, 12, 24, 48, 72):
                        step = int(checkpoint_hours * 4) - 1
                        if step >= len(pred_values):
                            continue
                        records.append({
                            'Growth scenario (%)': diagnostic_growth,
                            'Forecast hour': checkpoint_hours,
                            'Input peak (kW)': float(np.max(seed_values)),
                            'Input energy (kWh)': float(np.sum(seed_values) * 0.25),
                            'LSTM prediction (kW)': float(pred_values[step]),
                            'Baseline prediction (kW)': float(baseline_values[step]),
                            'Difference vs baseline (kW)': float(pred_values[step] - baseline_values[step]),
                        })
                diagnostics = pd.DataFrame(records)
                st.markdown('<div class="ov-section"><strong>📊 Scenario response by forecast horizon</strong><small>Detailed model diagnostics</small></div>', unsafe_allow_html=True)
                st.caption('Positive values mean higher predicted demand than the 0% baseline; negative values mean lower demand. Values near zero indicate convergence.')
                st.dataframe(diagnostics.style.set_properties(**{'color':'#112B4A','background-color':'#E7F1FC','font-weight':'700'}).set_table_styles([{'selector':'th','props':[('color','#102D50'),('background-color','#BEDCF9'),('font-weight','800')]}]).format({
                    'Forecast hour': '{:.2f}', 'Input peak (kW)': '{:.3f}',
                    'Input energy (kWh)': '{:.2f}', 'LSTM prediction (kW)': '{:.4f}',
                    'Baseline prediction (kW)': '{:.4f}',
                    'Difference vs baseline (kW)': '{:+.4f}',
                }), hide_index=True, use_container_width=True)
                diag_fig = go.Figure()
                diag_colors = {20: '#087F8C', 50: '#7C3AED', 100: '#B45309'}
                for diagnostic_growth in (20, 50, 100):
                    sub = diagnostics[diagnostics['Growth scenario (%)'] == diagnostic_growth]
                    diag_fig.add_trace(go.Scatter(
                        x=sub['Forecast hour'], y=sub['Difference vs baseline (kW)'],
                        mode='lines+markers', name=f'{diagnostic_growth}% vs 0%',
                        line=dict(color=diag_colors[diagnostic_growth], width=3),
                        marker=dict(size=9),
                        hovertemplate='Hour %{x:g}<br>Difference %{y:+.4f} kW<extra>' + str(diagnostic_growth) + '% growth</extra>'))
                diag_fig.add_hline(y=0, line_color='#334155', line_width=1)
                diag_fig.update_layout(
                    title=dict(text='How quickly does the model forget fleet growth?', font=dict(size=21, color='#112D54')),
                    xaxis_title='Hours after prediction start',
                    yaxis_title='Difference from 0% scenario (kW)',
                    height=400, font=dict(size=15, color='#102A4A'),
                    paper_bgcolor='#F3F8FF', plot_bgcolor='#EAF2FC',
                    xaxis=dict(tickfont=dict(color='#142E50', size=14), gridcolor='#B9CEE7', zeroline=False),
                    yaxis=dict(tickfont=dict(color='#142E50', size=14), gridcolor='#B9CEE7', zeroline=False),
                    legend=dict(orientation='h', y=1.13, font=dict(color='#0F172A')),
                    margin=dict(l=85, r=30, t=105, b=70), hovermode='x unified')
                st.plotly_chart(diag_fig, use_container_width=True)
                first_delta = float(diagnostic_predictions[100].iloc[0] - base_pred.iloc[0])
                first_reference = max(float(base_pred.iloc[0]), 0.1)
                early = first_delta
                late = float(diagnostic_predictions[100].iloc[95] - base_pred.iloc[95]) if len(base_pred) >= 96 else None
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric('Initial growth response', f'{first_delta:+.3f} kW', help='100% growth minus baseline at the first 15-minute prediction')
                with col_b:
                    st.metric('24-hour growth response', f'{late:+.3f} kW' if late is not None else 'Not available', help='100% growth minus baseline at the 24-hour prediction')
                with col_c:
                    if late is not None and abs(early) > 0.000001:
                        fade_pct = 100 * (1 - abs(late) / abs(early))
                        st.metric('Response reduction', f'{fade_pct:.0f}%' if fade_pct >= 0 else 'Increased', help='Reduction in absolute response from first prediction to hour 24')
                    else:
                        st.metric('Response reduction', 'Not available')
                if abs(first_delta) < max(0.1, first_reference * 0.05):
                    st.markdown('<div class="opt-diag-card amber"><strong>⚠️ Low immediate sensitivity</strong><br><span>The 100% growth input barely changes the first prediction. Review input scaling, training data and model fit before using this model for fleet-growth planning.</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="opt-diag-card green"><strong>✓ The model detects growth initially</strong><br><span>The first prediction responds to the growth-adjusted input. The chart shows whether this effect persists through the recursive forecast.</span></div>', unsafe_allow_html=True)
                if late is not None and abs(early) > 0.1 and abs(late) < abs(early) * 0.2:
                    st.markdown('<div class="opt-diag-card amber"><strong>⚠️ Growth signal fades by hour 24</strong><br><span>The 100% growth scenario difference has reduced by at least 80%. Recursive convergence is a likely contributor. Do not interpret this as evidence that real future fleet growth has no effect on electricity demand.</span></div>', unsafe_allow_html=True)
                st.markdown('<div class="opt-diag-card"><strong>Why this matters</strong><br><span>If scenario lines converge, the LSTM alone may be unsuitable for estimating long-term fleet expansion. Use a separate growth-adjusted planning projection alongside the LSTM forecast, and validate accuracy using held-out measured data.</span></div>', unsafe_allow_html=True)
                st.download_button('Download model diagnostics CSV', diagnostics.to_csv(index=False),
                                   file_name=f'{_slug(name)}_lstm_diagnostics_{future_date}.csv',
                                   mime='text/csv', key=f'download_diagnostics_{_slug(name)}')
                st.caption('This is a sensitivity test, not an accuracy or uncertainty assessment. '
                           'Validate forecast errors on held-out real observations before operational use.')
            st.caption('The future-date predictions are conditional on simulated historical-pattern inputs. '
                       'They have not been validated against actual demand on this date.')
        else:
            st.caption('The saved LSTM forecasts from the latest available actual reading; '
                       'it does not automatically receive live telemetry.')
    except Exception as exc:
        st.error(f'Forecast unavailable: {exc}')


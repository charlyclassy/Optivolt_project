"""Experimental direct multi-output demand model. Does not alter production LSTM.

Fleet and charger inputs are user assumptions, not observed UKPN metadata.
Train/test split is chronological. Synthetic augmentation is training-only.
"""
from __future__ import annotations
import hashlib
import io
import numpy as np
import pandas as pd
import streamlit as st

LOOKBACK = 96
HORIZON = 96  # direct 24h, 15-minute steps
STEP = 4  # hourly training origins for efficiency


def _profile(ts, vehicles, charger_kw, utilisation, growth):
    """Known-in-advance calendar and user-defined operational assumptions."""
    t = pd.DatetimeIndex(ts)
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60
    dow = t.dayofweek.to_numpy()
    month = t.month.to_numpy()
    # Deterministic overnight charging preference, not measured charger activity.
    overnight = np.where((hour >= 19) | (hour < 7), 1.0, 0.25)
    weekend = np.where(dow >= 5, 0.65, 1.0)
    available_kw = vehicles * charger_kw * utilisation * overnight * weekend
    return np.column_stack([
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7),
        np.sin(2*np.pi*month/12), np.cos(2*np.pi*month/12),
        np.full(len(t), growth/100), available_kw / 100,
    ]).astype('float32')


def _clean(raw):
    d = raw[['timestamp','demand_kw']].copy()
    d['timestamp'] = pd.to_datetime(d.timestamp)
    d = d.drop_duplicates('timestamp').set_index('timestamp').sort_index()
    d = d.resample('15min').mean()
    # Only short gaps are interpolated; longer gaps remain missing and windows are excluded.
    d['demand_kw'] = d['demand_kw'].interpolate(limit=4, limit_area='inside')
    return d['demand_kw'].clip(lower=0)


def _windows(series, max_train=2400, max_test=300, seed=42):
    n = len(series)
    if n < LOOKBACK + HORIZON + 200:
        raise ValueError('Insufficient continuous 15-minute history for a held-out evaluation.')
    boundary = int(n * .8)
    # Purge 24 hours around split; target never crosses training boundary.
    train_pos = np.arange(LOOKBACK, boundary-HORIZON, STEP)
    test_pos = np.arange(boundary+LOOKBACK, n-HORIZON+1, STEP)
    rng = np.random.default_rng(seed)
    if len(train_pos) > max_train: train_pos = np.sort(rng.choice(train_pos, max_train, replace=False))
    if len(test_pos) > max_test: test_pos = np.sort(rng.choice(test_pos, max_test, replace=False))
    a = series.to_numpy(dtype='float32')
    train_pos = [p for p in train_pos if np.isfinite(a[p-LOOKBACK:p+HORIZON]).all()]
    test_pos = [p for p in test_pos if np.isfinite(a[p-LOOKBACK:p+HORIZON]).all()]
    if len(train_pos) < 80 or len(test_pos) < 15:
        raise ValueError('Not enough complete windows after filtering missing data.')
    return a, series.index, train_pos, test_pos


def _make(a, index, positions, scale, vehicles, charger_kw, utilisation, augment=False):
    rng = np.random.default_rng(31)
    x_hist, x_future, y = [], [], []
    for p in positions:
        hist = a[p-LOOKBACK:p] / scale
        target = a[p:p+HORIZON] / scale
        # Synthetic scenarios only augment training, never held-out observations.
        growth = float(rng.choice([0, 20, 50, 100])) if augment else 0.
        multiplier = 1 + growth / 100
        hist = hist * multiplier
        target = target * multiplier
        feat = _profile(index[p:p+HORIZON], vehicles, charger_kw, utilisation, growth)
        x_hist.append(hist[:,None]); x_future.append(feat); y.append(target)
    return (np.asarray(x_hist,dtype='float32'), np.asarray(x_future,dtype='float32')), np.asarray(y,dtype='float32')


def train_and_backtest(raw, vehicles=12, charger_kw=7.0, utilisation=.35, epochs=5,
                       progress_callback=None, improve=False):
    import tensorflow as tf
    from tensorflow import keras
    tf.keras.utils.set_random_seed(42)
    series = _clean(raw)
    a, index, train_pos, test_pos = _windows(series)
    # Fit scale on training data only.
    scale = max(float(np.nanpercentile(a[:int(len(a)*.8)],95)),1.0)
    xtrain, ytrain = _make(a,index,train_pos,scale,vehicles,charger_kw,utilisation,True)
    xtest, ytest = _make(a,index,test_pos,scale,vehicles,charger_kw,utilisation,False)
    # Candidate selection uses ONLY the chronological training/validation portion.
    # The untouched held-out test portion is evaluated once, after selection.
    cut = max(1, int(len(ytrain) * .85))
    configs = ([(32, 1e-3), (64, 5e-4), (48, 3e-4)] if improve else [(32, 1e-3)])
    best_model, best_loss, best_config = None, float('inf'), None
    total = len(configs) * int(epochs)

    for attempt, (units, learning_rate) in enumerate(configs, 1):
        tf.keras.utils.set_random_seed(42 + attempt)
        hist_input = keras.Input((LOOKBACK, 1), name='demand_history')
        future_input = keras.Input((HORIZON, 8), name='known_future_features')
        h = keras.layers.LSTM(units)(hist_input)
        f = keras.layers.Dense(32, activation='relu')(keras.layers.Flatten()(future_input))
        merged = keras.layers.Concatenate()([h, f])
        merged = keras.layers.Dense(96, activation='relu')(merged)
        out = keras.layers.Dense(HORIZON, activation='softplus')(merged)
        candidate = keras.Model([hist_input, future_input], out)
        candidate.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse')

        class ReportProgress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress_callback is not None:
                    progress_callback(attempt, len(configs), epoch + 1, int(epochs),
                                      (logs or {}).get('val_loss'))

        history = candidate.fit(
            [xtrain[0][:cut], xtrain[1][:cut]], ytrain[:cut],
            validation_data=([xtrain[0][cut:], xtrain[1][cut:]], ytrain[cut:]),
            epochs=int(epochs), batch_size=32, verbose=0,
            callbacks=[ReportProgress(), keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True)])
        loss = float(min(history.history['val_loss']))
        if loss < best_loss:
            best_loss, best_model, best_config = loss, candidate, {
                'lstm_units': units, 'learning_rate': learning_rate,
                'selected_validation_loss': round(loss, 6), 'attempts': len(configs)}
        if progress_callback is not None:
            progress_callback(attempt, len(configs), int(epochs), int(epochs), loss)

    model = best_model
    pred = model.predict(list(xtest),verbose=0)*scale
    truth = ytest*scale
    # Persistence: repeat last observed 24h. Strong, simple benchmark.
    persistence = xtest[0][:,:,0]*scale
    rows=[]
    for hours in (1,6,12,24):
        s=hours*4
        for label, vals in [('Newly Trained LSTM',pred),('Previous-day baseline',persistence)]:
            diff = vals[:,:s]-truth[:,:s]
            rows.append({'Horizon (hours)':hours,'Model':label,
                         'MAE (kW)':round(float(np.mean(np.abs(diff))),3),
                         'RMSE (kW)':round(float(np.sqrt(np.mean(diff**2))),3),
                         'NMAE (%)':round(float(100*np.mean(np.abs(diff))/max(np.mean(np.abs(truth[:,:s])),1e-6)),2),
                         'Test windows':len(test_pos)})
    return model,scale,series,pd.DataFrame(rows),{'train_windows':len(train_pos),'test_windows':len(test_pos),'test_start':str(index[test_pos[0]]),'test_end':str(index[test_pos[-1]+HORIZON-1]),
            'example_observed_kw':truth[0].round(3).tolist(),
            'example_predicted_kw':pred[0].round(3).tolist(),
            'example_start':str(index[test_pos[0]]), 'training_selection':best_config,
            'holdout_peak_mae_kw':round(float(np.mean(np.abs(pred.max(axis=1)-truth.max(axis=1)))),3),
            'holdout_peak_underprediction_pct':round(float(100*np.mean(pred.max(axis=1)<truth.max(axis=1))),1),
            'holdout_actual_peaks_kw':truth.max(axis=1).round(3).tolist(),
            'holdout_predicted_peaks_kw':pred.max(axis=1).round(3).tolist()}


def predict_scenario(model,scale,series,start,vehicles,charger_kw,utilisation,growth):
    ts = pd.Timestamp(start)
    # Planning seed from matching historical weekday/time-of-day median; never future measurements.
    # Self-contained historical seed: no dependency on future_simulation.py.
    # Prefer the median of complete matching weekdays; otherwise use complete days.
    history = series.dropna()
    days = []
    for day, group in history.groupby(history.index.normalize()):
        if len(group) != LOOKBACK:
            continue
        if not (group.index == pd.date_range(day, periods=LOOKBACK, freq='15min')).all():
            continue
        days.append((day, group.to_numpy(dtype='float32')))
    if not days:
        raise ValueError('No complete historical 24-hour days to seed the future forecast.')
    matching = [v for day, v in days if day.dayofweek == (ts - pd.Timedelta(days=1)).dayofweek]
    pool = matching if matching else [v for _, v in days]
    values = np.median(np.stack(pool[-8:]), axis=0).astype('float32')
    values *= (1 + growth / 100)
    future = pd.date_range(ts,periods=HORIZON,freq='15min')
    feat = _profile(future,vehicles,charger_kw,utilisation,growth)
    y = model.predict([values.reshape(1,LOOKBACK,1)/scale,feat.reshape(1,HORIZON,8)],verbose=0)[0]*scale
    return pd.Series(np.maximum(y,0),index=future,name=f'{growth}% scenario')


def render(name, raw):
    st.markdown('''<div style="background:linear-gradient(110deg,#102C63,#087F8C);color:white;padding:14px 18px;border-radius:13px;margin-top:10px">
    <div style="font-size:22px;font-weight:800;color:white">Growth-aware direct forecasting · Experimental</div>
    <div style="color:#E8F8FF">24-hour multi-output predictions, charging assumptions and chronological backtesting.</div></div>''',unsafe_allow_html=True)
    st.markdown('''<style>
    .ga-panel {background:linear-gradient(115deg,#E7F2FF,#E5F9F6);border:1px solid #B8D8EE;
      border-radius:14px;padding:12px 16px;margin:8px 0;color:#13385D}
    .ga-panel strong {color:#123C70}
    </style>''',unsafe_allow_html=True)
    status = 'Trained in this session' if st.session_state.get(f'ga_result_{name}') is not None else 'Not yet trained'
    st.markdown(f'''<div class="ga-panel"><strong>Model status:</strong> {status} &nbsp; | &nbsp;
    <strong>Architecture:</strong> 96-step direct output (24 hours) &nbsp; | &nbsp;
    <strong>Evaluation:</strong> chronological holdout + previous-day baseline</div>''',unsafe_allow_html=True)
    st.markdown('**Evidence of the four improvements**')
    st.markdown('''| Improvement | What is implemented | Where to verify |
|---|---|---|
| Direct multi-step LSTM | One forward pass produces 96 demand values | 24-hour scenario chart after training |
| Extra features | Calendar, assumed vehicles, charger rating, utilisation and charging schedule | Controls below and `known_future_features` input |
| Growth-aware training | Synthetic growth-adjusted training windows | Training configuration and scenario outputs |
| Historical backtesting | Chronological holdout with MAE/RMSE at 1, 6, 12 and 24 hours | Backtest table and CSV after training |''')
    st.caption('Experimental model: no verified fleet telemetry. Original saved LSTM models remain unchanged. The benchmark is previous-day persistence, not the original LSTM.')
    c1,c2,c3,c4=st.columns(4)
    with c1: vehicles=st.number_input('Vehicles',1,1000,12,key='ga_vehicles')
    with c2: charger=st.number_input('Charger rating (kW)',1.,350.,7.,key='ga_charger')
    with c3: utilisation=st.slider('Charging utilisation (%)',0,100,35,key='ga_util')/100
    with c4: epochs=st.selectbox('Training epochs',[3,5,10,20],index=1,key='ga_epochs')
    st.caption('The charging schedule is an illustrative overnight preference, not observed charger-session data. Training augmentation scales historical demand; it cannot establish real-world fleet-growth causality.')
    if st.button('Train experimental model and backtest',type='primary',key='ga_train'):
        try:
            with st.spinner('Training direct 24-hour model and testing on held-out chronological observations…'):
                result=train_and_backtest(raw,vehicles,charger,utilisation,epochs)
            st.session_state[f'ga_result_{name}']=result
        except Exception as exc:
            st.error(f'Experimental training unavailable: {exc}')
    result=st.session_state.get(f'ga_result_{name}')
    if result is None:
        st.info('Training has not been run in this session. Select the assumptions above and click Train experimental model and backtest to generate actual evaluation evidence.')
        return
    model,scale,series,metrics,details=result
    st.markdown('### Historical backtesting · measured holdout')
    st.dataframe(metrics,hide_index=True,use_container_width=True)
    st.caption(f"Training windows: {details['train_windows']} · Test windows: {details['test_windows']} · Test period: {details['test_start']} to {details['test_end']}. No synthetic augmentation in the test set.")
    st.download_button('Download backtest CSV',metrics.to_csv(index=False),file_name='growth_aware_backtest.csv',mime='text/csv')
    st.markdown('### Direct 24-hour growth scenarios')
    d,g=st.columns(2)
    with d: date=st.date_input('Prediction start date',value=pd.Timestamp.today().date()+pd.Timedelta(days=1),key='ga_date')
    with g: choices=st.multiselect('Fleet-growth assumptions (%)',[0,20,50,100,150],default=[0,20,50,100],key='ga_scenarios')
    if st.button('Compare direct forecasts',key='ga_compare') and choices:
        try:
            preds={pct:predict_scenario(model,scale,series,date,vehicles,charger,utilisation,pct) for pct in choices}
            st.session_state[f'ga_predictions_{name}']=preds
        except Exception as exc: st.error(f'Scenario prediction unavailable: {exc}')
    preds=st.session_state.get(f'ga_predictions_{name}')
    if preds:
        import plotly.graph_objects as go
        fig=go.Figure()
        for pct,p in preds.items(): fig.add_trace(go.Scatter(x=p.index,y=p.values,name=f'{pct}% growth',mode='lines',line={'width':3}))
        fig.update_layout(template='plotly_white',height=390,margin=dict(l=30,r=20,t=35,b=25),
                          xaxis_title='Date and time',yaxis_title='Predicted demand (kW)',font={'color':'#102B4E','size':14},legend={'orientation':'h','y':1.14})
        st.plotly_chart(fig,use_container_width=True)
        summary=pd.DataFrame([{'Growth (%)':k,'Peak (kW)':round(float(v.max()),2),'Energy (kWh)':round(float(v.sum()*.25),2)} for k,v in preds.items()])
        st.dataframe(summary,hide_index=True,use_container_width=True)
        st.caption('These are scenario-conditioned model outputs, not validated predictions of future fleet growth. Compare with historical backtest before use.')

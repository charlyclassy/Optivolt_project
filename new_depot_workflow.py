"""New-depot onboarding: choose transferable saved LSTM or train a dedicated model."""
import hashlib
from datetime import date, time, datetime
import numpy as np
import pandas as pd
import streamlit as st
import depot_updates as dep


def compatibility(frame, model_name):
    """Check the actual inference contract, not just the uploaded CSV column names."""
    s = dep._regular(frame)
    x, y, _ = dep._windows(s)
    model, scalers = dep._load_model(model_name)
    shape = model.input_shape
    if isinstance(shape, list):
        raise ValueError('Multi-input saved models are not supported by this demand-only workflow.')
    if len(shape) != 3 or shape[1] != 96 or shape[2] != 1:
        raise ValueError(f'Saved model expects {shape}; this workflow supplies (batch, 96, 1).')
    if len(model.output_shape) != 2 or model.output_shape[-1] != 1:
        raise ValueError(f'Unexpected model output shape: {model.output_shape}')
    if not np.all(np.isfinite(s.to_numpy())) or not np.all(np.isfinite(scalers)):
        raise ValueError('Non-finite data or saved scaling parameters.')
    return model, scalers, x, y, s


def _chart(actual, pred, title, capacity=None, forecast=False):
    """Readable, consistent Optivolt plots for evaluation and future demand."""
    import plotly.graph_objects as go
    fig = go.Figure()
    actual = actual.sort_index()
    pred = pred.sort_index()
    if len(actual):
        fig.add_trace(go.Scatter(
            x=actual.index, y=actual.to_numpy(), mode='lines',
            name='Observed demand (uploaded data)',
            line=dict(color='#2563eb', width=2.7),
            hovertemplate='%{x|%d %b %Y, %H:%M}<br>Observed: %{y:.2f} kW<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=pred.index, y=pred.to_numpy(), mode='lines',
        name='Forecast demand (LSTM)' if forecast else 'Saved LSTM prediction',
        line=dict(color='#0d9488', width=3),
        hovertemplate='%{x|%d %b %Y, %H:%M}<br>Predicted: %{y:.2f} kW<extra></extra>'))
    if forecast and capacity is not None and float(capacity) > 0:
        for threshold, label, color, dash in (
            (float(capacity), f'Grid capacity ({capacity:g} kW)', '#dc2626', 'dash'),
            (float(capacity)*0.8, '80% capacity warning', '#d97706', 'dot')):
            fig.add_hline(y=threshold, line_color=color, line_width=2, line_dash=dash)
            fig.add_annotation(x=0.995, xref='paper', y=threshold, yref='y',
                text=label, showarrow=False, xanchor='right', yanchor='bottom',
                font=dict(size=12, color=color), bgcolor='rgba(255,255,255,0.92)',
                borderpad=3)
    fig.update_layout(
        title=dict(text=title, x=0.015, xanchor='left',
                   font=dict(size=19, color='#152238')),
        height=490 if forecast else 440,
        margin=dict(l=70, r=30, t=125, b=75),
        font=dict(family='Arial, sans-serif', size=14, color='#24344b'),
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        legend=dict(orientation='h', x=0, y=1.12, xanchor='left', yanchor='bottom',
                    font=dict(size=13), bgcolor='rgba(255,255,255,0)'),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#ffffff', font_size=13, font_color='#152238'))
    fig.update_xaxes(title_text='Date and time', title_font=dict(size=14),
        tickfont=dict(size=12), tickformat='%d %b\n%H:%M',
        showgrid=True, gridcolor='#e9eef5', showline=True, linecolor='#cbd5e1',
        automargin=True, nticks=8)
    fig.update_yaxes(title_text='Electricity demand (kW)', title_font=dict(size=14),
        tickfont=dict(size=13), tickformat='.1f', rangemode='tozero',
        showgrid=True, gridcolor='#e4eaf2', zeroline=False, automargin=True)
    st.plotly_chart(fig, use_container_width=True, config={
        'displaylogo': False, 'toImageButtonOptions': {'format':'png','scale':2}})



def _evaluation_dashboard(actual, predicted, capacity, label, baseline=None, historical_note=None):
    """Present transparent model metrics computed from the evaluated observations."""
    a=np.asarray(actual,dtype=float).reshape(-1)
    p=np.asarray(predicted,dtype=float).reshape(-1)
    valid=np.isfinite(a)&np.isfinite(p)
    a,p=a[valid],p[valid]
    if not len(a):
        st.warning('No valid observations available for evaluation.')
        return
    error=np.abs(a-p)
    mae=float(error.mean())
    rmse=float(np.sqrt(np.mean((a-p)**2)))
    within1=float(np.mean(error<=1)*100)
    within3=float(np.mean(error<=3)*100)
    over3=float(np.mean(error>3)*100)
    r2=1-float(np.sum((a-p)**2)/np.sum((a-a.mean())**2)) if np.var(a)>0 else None
    st.markdown(f'#### {label} · Model performance')
    cols=st.columns(4)
    for col,caption,value in zip(cols,['Mean absolute error','Root mean squared error','Predictions within ±1 kW','Predictions within ±3 kW'],
                                  [f'{mae:.2f} kW',f'{rmse:.2f} kW',f'{within1:.1f}%',f'{within3:.1f}%']):
        col.metric(caption,value)
    st.markdown('**Prediction error tolerance**')
    for title,pct in [('Within ±1 kW',within1),('Within ±3 kW',within3),('Errors above 3 kW',over3)]:
        left,right=st.columns([4,1])
        left.write(title)
        right.markdown(f'**{pct:.1f}%**')
        st.progress(min(100,max(0,int(round(pct)))))
    if r2 is not None:
        st.caption(f'R² score: {r2:.3f} · Evaluated observations: {len(a):,}')
    conclusion=f'The model has a mean absolute error of {mae:.2f} kW; {within1:.1f}% of predictions are within ±1 kW and {within3:.1f}% are within ±3 kW.'
    if baseline is not None:
        b=np.asarray(baseline,dtype=float).reshape(-1)[valid]
        if np.all(np.isfinite(b)):
            base_mae=float(np.mean(np.abs(a-b)))
            base_rmse=float(np.sqrt(np.mean((a-b)**2)))
            st.markdown('**Comparison with previous-day, same-time baseline**')
            st.dataframe(pd.DataFrame([
                {'Metric':'Mean absolute error (MAE)','Previous-day baseline':f'{base_mae:.2f} kW','LSTM':f'{mae:.2f} kW'},
                {'Metric':'Root mean squared error (RMSE)','Previous-day baseline':f'{base_rmse:.2f} kW','LSTM':f'{rmse:.2f} kW'},
            ]),hide_index=True,use_container_width=True)
            if base_mae>0:
                improvement=100*(base_mae-mae)/base_mae
                st.metric('MAE improvement vs previous-day baseline',f'{improvement:.1f}%')
                conclusion+= (' It outperforms the previous-day baseline on this evaluation.' if improvement>0 else ' It does not outperform the previous-day baseline on this evaluation.')
    if historical_note:
        st.caption(historical_note)
    st.success('Historical evaluation summary: '+conclusion)
    st.caption('These metrics measure historical one-step-ahead predictions, not accuracy of a future multi-day forecast. Strong performance should be confirmed on observations not used for model training.')


def _forecast(model, scalers, series, hours, progress=None):
    """Recursive 15-minute prediction with visible progress and an explicit result."""
    history = series.iloc[-96:]
    if len(history) != 96 or not (history.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all():
        raise ValueError('The latest 96 readings must be consecutive 15-minute observations.')
    window = history.to_numpy(np.float32).copy()
    xm,xs,ym,ys = scalers
    out=[]
    count=int(hours*4)
    for step in range(count):
        z = ((window-xm)/xs).reshape(1,96,1).astype(np.float32)
        next_kw = max(0., float(np.asarray(model(z,training=False)).reshape(-1)[0])*ys+ym)
        if not np.isfinite(next_kw):
            raise ValueError('Model returned a non-finite forecast value. Check saved scaling parameters.')
        out.append(next_kw)
        window[:-1] = window[1:]
        window[-1] = next_kw
        if progress is not None and ((step+1)%4==0 or step+1==count):
            progress.progress((step+1)/count, text=f'Forecasting: {step+1}/{count} intervals ({(step+1)/4:g}/{hours:g} hours)')
    times=pd.date_range(history.index[-1]+pd.Timedelta(minutes=15),periods=len(out),freq='15min')
    return pd.Series(out,index=times)



def _historical_analogue_forecast(model, scalers, series, start, end, progress=None):
    """Date-selected scenario: independently model each target day from a historical analogue.

    This does not pretend the model has observed the missing months. Each target day
    uses a real consecutive 96-point history window from the same weekday, preferring
    the closest calendar season; the model then predicts that day's demand recursively.
    """
    if series.empty:
        raise ValueError('No historical observations available.')
    series = series.sort_index()
    requested = pd.date_range(start, end - pd.Timedelta(minutes=15), freq='15min')
    days = pd.DatetimeIndex(requested.normalize().unique())
    if len(days) == 0:
        raise ValueError('Choose a non-empty forecast period.')
    # Candidate histories must be consecutive, finite and exactly 96 samples long.
    candidates = []
    values = series.to_numpy(dtype=float)
    times = series.index
    for end_idx in range(95, len(series)):
        if (times[end_idx] - times[end_idx - 95] == pd.Timedelta(minutes=15*95)
                and np.isfinite(values[end_idx-95:end_idx+1]).all()):
            # Historical window ending at 23:45 provides an analogue for next day.
            if times[end_idx].hour == 23 and times[end_idx].minute == 45:
                candidates.append(end_idx)
    if not candidates:
        raise ValueError('No complete historical calendar-day input windows were found.')
    output = {}
    total_steps = len(days) * 96
    completed = 0
    for target_day in days:
        # Prefer the same weekday and a similar time of year. If unavailable,
        # use the nearest seasonal analogue irrespective of weekday.
        target_doy = target_day.dayofyear
        def rank(i):
            analogue_day = times[i].normalize() + pd.Timedelta(days=1)
            weekday_penalty = 0 if analogue_day.weekday() == target_day.weekday() else 1000
            seasonal = abs(analogue_day.dayofyear - target_doy)
            seasonal = min(seasonal, 366 - seasonal)
            return (weekday_penalty + seasonal, abs((target_day-analogue_day).days))
        idx = min(candidates, key=rank)
        window = values[idx-95:idx+1].astype(np.float32).copy()
        xm, xs, ym, ys = scalers
        if xs == 0 or ys == 0:
            raise ValueError('Saved model has invalid zero scaling values.')
        for quarter in range(96):
            z = ((window-xm)/xs).reshape(1, 96, 1).astype(np.float32)
            value = max(0.0, float(np.asarray(model(z, training=False)).reshape(-1)[0])*ys + ym)
            if not np.isfinite(value):
                raise ValueError('The model produced a non-finite scenario value.')
            output[target_day + pd.Timedelta(minutes=15*quarter)] = value
            window[:-1] = window[1:]
            window[-1] = value
            completed += 1
            if progress is not None and (completed % 8 == 0 or completed == total_steps):
                progress.progress(min(1.0, completed / total_steps),
                    text=f'Generating future demand forecast: {completed}/{total_steps} intervals')
    return pd.Series([output[t] for t in requested], index=requested, dtype=float)

def _forecast_period(key, frame):
    """Return selected 2026 range; end is exclusive, at 15-minute resolution."""
    st.markdown('##### Select forecast period (2026)')
    c1, c2 = st.columns(2)
    with c1:
        from_date = st.date_input('From date', value=date(2026, 11, 9),
                                  min_value=date(2026, 1, 1), max_value=date(2026, 12, 31), key=key+'_from_date')
        from_time = st.time_input('From time', value=time(0, 0), step=900, key=key+'_from_time')
    with c2:
        to_date = st.date_input('To date', value=date(2026, 11, 10),
                                min_value=date(2026, 1, 1), max_value=date(2026, 12, 31), key=key+'_to_date')
        to_time = st.time_input('To time (end exclusive)', value=time(0, 0), step=900, key=key+'_to_time')
    start = pd.Timestamp(datetime.combine(from_date, from_time))
    end = pd.Timestamp(datetime.combine(to_date, to_time))
    if end <= start:
        st.error('The To date/time must be later than the From date/time.')
        return None
    if start.minute % 15 or end.minute % 15 or start.second or end.second:
        st.error('Choose times on 15-minute boundaries (00, 15, 30 or 45).')
        return None
    count = int((end-start)/pd.Timedelta(minutes=15))
    if count > 288:
        st.error('Select a period of up to 72 hours (288 intervals) for this recursive LSTM.')
        return None
    latest = dep._regular(frame).index[-1]
    st.caption(f'Selected: {start:%d %b %Y %H:%M} to {end:%d %b %Y %H:%M} · {count} intervals · end exclusive')
    if start > latest + pd.Timedelta(minutes=15):
        st.caption(f'Forecast using available historical readings (latest: {latest:%d %b %Y}). '
                   'For each selected day, the LSTM uses a historical analogue with a matching weekday where possible. '
                   'Forecast estimates are based on historical patterns, not observations from the selected future dates.')
        scenario = True
    else:
        scenario = False
        if start != latest + pd.Timedelta(minutes=15):
            st.error('For a continuous forecast, From must be exactly 15 minutes after the latest reading.')
            return None
    return start, end, count, scenario


def _forecast_results(forecast, history, capacity, name, download_name, scenario=False):
    st.markdown('#### Forecast results')
    if scenario:
        st.caption('Forecast generated using the saved LSTM and available historical readings.')
    peak=float(forecast.max())
    energy=float(forecast.sum()*.25)
    exceed=int((forecast>capacity).sum())
    a,b,c,d=st.columns(4)
    a.metric('Predicted peak', f'{peak:.2f} kW')
    b.metric('Grid headroom', f'{capacity-peak:+.2f} kW')
    c.metric('Forecast energy', f'{energy:.1f} kWh')
    d.metric('Capacity exceedances', f'{exceed} intervals')
    if exceed:
        st.error(f'Capacity exceeded in {exceed} of {len(forecast)} intervals ({exceed*15} minutes).')
    else:
        st.success(f'No forecast intervals exceed the configured grid capacity ({capacity:g} kW).')
    # Do not connect a 2025 history trace visually to a 2026 scenario projection.
    shown_history = pd.Series(dtype=float) if scenario else history.iloc[-96:]
    _chart(shown_history, forecast, f'{name} | '+(f'Future demand forecast — {forecast.index[0]:%d %b %Y}' if scenario else 'Observed demand and future forecast'), capacity, forecast=True)
    st.caption(f'Period: {forecast.index[0]:%d %b %Y %H:%M} to '
               f'{forecast.index[-1]+pd.Timedelta(minutes=15):%d %b %Y %H:%M} (end exclusive)')
    with st.expander('Forecast values and capacity assessment'):
        table=pd.DataFrame({'timestamp':forecast.index,'predicted_demand_kw':forecast.values.round(4),
                            'above_capacity':['Exceeded' if v>capacity else 'Within limit' for v in forecast.values]})
        st.dataframe(table,hide_index=True,use_container_width=True,height=390,
                     column_config={'timestamp':st.column_config.DatetimeColumn('Date and time',format='DD MMM YYYY, HH:mm'), 'predicted_demand_kw':st.column_config.NumberColumn('Predicted demand (kW)',format='%.2f'), 'above_capacity':st.column_config.TextColumn('Grid capacity status')})
    import forecast_results_store
    try:
        forecast_results_store.save(name, forecast, capacity, 'Depot-specific LSTM' if 'specific' in download_name else 'Saved LSTM')
    except Exception as exc:
        st.warning(f'Forecast displayed but could not be saved for other pages: {exc}')
    st.download_button('Download forecast CSV', table.to_csv(index=False),
        file_name=download_name,mime='text/csv')


def _run_forecast(model, scalers, frame, period, cache_key):
    """Run bounded 15-minute prediction with visible progress and safe date labels."""
    start, end, count, scenario = period
    hours=count/4
    stage=st.empty()
    progress=st.progress(0,text='Step 1/4 · Validating recent observations')
    try:
        series=dep._regular(frame)
        if len(series)<96:
            raise ValueError('At least 96 consecutive recent readings are required.')
        stage.info('Step 1/4 · Data prepared. Checking model inputs.')
        progress.progress(.05,text='Step 2/4 · Checking saved model inputs')
        if not np.all(np.isfinite(np.asarray(scalers,dtype=float))):
            raise ValueError('Invalid model scaling parameters.')
        stage.info('Step 3/4 · Generating '+('scenario' if scenario else 'forecast')+' intervals.')
        if scenario:
            forecast=_historical_analogue_forecast(model,scalers,series,start,end,progress)
        else:
            forecast=_forecast(model,scalers,series,hours,progress)
        if len(forecast) != count or not forecast.index.equals(pd.date_range(start, periods=count, freq='15min')):
            raise ValueError('Forecast timestamps do not match the selected date range.')
        stage.info('Step 4/4 · Calculating capacity results and preparing chart.')
        st.session_state[cache_key]={'start':str(start),'end':str(end),'scenario':scenario,
                                     'times':[str(t) for t in forecast.index], 'values':forecast.tolist()}
        progress.progress(1.0,text='Completed · Results ready')
        stage.success('Processing completed. Results are shown below.')
    except Exception as exc:
        progress.empty()
        stage.error(f'Forecast could not be generated: {exc}')


def _cached_forecast_result(cache_key, period):
    result=st.session_state.get(cache_key)
    if result and result['start']==str(period[0]) and result['end']==str(period[1]) and result['scenario']==period[3]:
        return pd.Series(result['values'],index=pd.to_datetime(result['times']))
    return None

def render(name, frame, capacity, file_hash):
    st.markdown('### 2 · Choose forecasting approach')
    path=st.radio('Model approach', ['Use an existing saved LSTM', 'Train a new depot-specific LSTM'],
                  horizontal=True,key='new_depot_model_approach')
    if path.startswith('Use'):
        st.caption('Compatibility → Evaluate on uploaded observations → Predict. No training or model replacement.')
        sources=[d for d in dep.visible_depots() if all(p.exists() for p in dep.active_paths(d))]
        if not sources:
            st.warning('No saved LSTM models are available. Choose Train a new depot-specific LSTM.')
            return
        source=st.selectbox('Saved LSTM to test',list(dict.fromkeys(sources)),key='new_depot_source_model')
        key=f'transfer_{hashlib.sha256((name+file_hash+source+str(dep.model_revision(source))).encode()).hexdigest()[:16]}'
        if st.button('Check model and data compatibility',type='primary'):
            with st.spinner('Step 1/2 · Validating uploaded readings and checking saved model inputs'):
              try:
                model,scalers,x,y,series=compatibility(frame,source)
                st.session_state[key]={'compatible':True,'evaluated':False}
              except Exception as exc:
                st.session_state[key]={'compatible':False,'reason':str(exc)}
        status=st.session_state.get(key)
        if status and not status['compatible']:
            st.error('Not compatible: '+status['reason'])
            return
        if not status:
            st.info('Run the compatibility check first. It verifies timestamps, continuous 24-hour input windows, model input/output shape and scaling parameters.')
            return
        st.success('Input format and saved-model interface are compatible. This does not guarantee good predictive accuracy.')
        try:
            series_checked = dep._regular(frame)
            windows_checked, _, _ = dep._windows(series_checked)
            st.caption(f'{len(series_checked):,} readings · {len(windows_checked):,} valid 24-hour LSTM input windows')
        except Exception as exc:
            st.warning(f'Window diagnostics unavailable: {exc}')
        if st.button('Evaluate saved LSTM on uploaded demand'):
            with st.spinner('Evaluating saved LSTM · preparing windows, predicting and calculating metrics'):
              try:
                model,scalers,x,y,series=compatibility(frame,source)
                predictions=dep._predict(model,scalers,x)
                metrics=dep._scores(y,predictions,capacity)
                metrics['NMAE (% of capacity)']=round(100*metrics['MAE (kW)']/capacity,1)
                baseline=x[:,0]  # same time on previous day (96 x 15 minutes earlier)
                metrics['Previous-day same-time baseline MAE (kW)']=round(float(np.mean(np.abs(y-baseline))),3)
                # A chronological evaluation on uploaded observations; model may have seen overlapping dates if reused.
                status['evaluated']=True
                status['metrics']=metrics
                status['evaluation_actual']=y.tolist()
                status['evaluation_pred']=predictions.tolist()
                status['evaluation_baseline']=baseline.tolist()
                status['actual']=y[-min(192,len(y)):].tolist()
                status['pred']=predictions[-min(192,len(y)):].tolist()
                status['times']=[str(t) for t in series.index[-len(y):][-min(192,len(y)):]]
              except Exception as exc:
                st.error(f'Evaluation failed: {exc}')
        if status.get('evaluated'):
            st.markdown('#### Model evaluation · uploaded historical demand')
            st.caption('Blue: measured readings from the uploaded dataset. Teal: saved LSTM estimates for those same historical timestamps. These are not 2026 measurements.')
            _evaluation_dashboard(status.get('evaluation_actual',status['actual']),status.get('evaluation_pred',status['pred']),capacity,'Saved LSTM',baseline=status.get('evaluation_baseline'),historical_note='Evaluation uses uploaded historical readings. Confirm that these dates were not included in the saved model’s training data before treating this as an independent test.')
            idx=pd.to_datetime(status['times'])
            _chart(pd.Series(status['actual'],index=idx),pd.Series(status['pred'],index=idx),f'Model evaluation | Actual vs predicted demand — {idx.min():%b %Y}')
            st.markdown('#### 3 · Forecast new depot demand')
            period=_forecast_period('nd_saved',frame)
            forecast_key=key+'_forecast'
            if st.button('Generate forecast with saved LSTM',type='primary',disabled=period is None):
                try:
                    with st.spinner('Loading saved LSTM and preparing forecast'):
                        model,scalers,_,_,series=compatibility(frame,source)
                    _run_forecast(model,scalers,frame,period,forecast_key)
                except Exception as exc:
                    st.error(f'Forecast could not be generated: {exc}')
            forecast=_cached_forecast_result(forecast_key,period) if period else None
            if forecast is not None:
                _forecast_results(forecast,dep._regular(frame),capacity,name,'new_depot_forecast.csv',period[3])
    else:
        st.caption('Train → Validate → Activate → Predict. The model is specific to this depot; existing UKPN models remain unchanged.')
        c1,c2,c3=st.columns(3)
        epochs=c1.number_input('Maximum epochs',min_value=1,max_value=150,value=20,key='nd_epochs')
        batch=c2.selectbox('Batch size',[16,32,64,128,256],index=2,key='nd_batch')
        lr=c3.selectbox('Learning rate',[0.0001,0.0003,0.001,0.003],index=2,key='nd_lr')
        c4,c5=st.columns(2)
        patience=c4.number_input('Early-stopping patience',min_value=1,max_value=30,value=5,key='nd_patience')
        lookback=c5.selectbox('Input history',["24 hours (96 readings)"],key='nd_lookback')
        st.caption('Chronological split: 70% train / 15% validation / 15% held-out test. Scalers fitted on training data only.')
        st.info('A separate model will be created for this depot. Existing UKPN models will not be overwritten.')
        if st.button('Train depot-specific LSTM',type='primary'):
            if len(frame)<1000:
                st.error('At least 1,000 readings are required for the chronological training and test split.')
            else:
                try:
                    st.info('Step 1/4 · Checking dataset and preparing chronological training, validation and test sets')
                    progress=st.progress(0,text='Step 2/4 · Training LSTM · preparing model')
                    result=dep.train_candidate(name,frame,capacity,int(epochs),progress,batch_size=int(batch),learning_rate=float(lr),patience=int(patience))
                    st.info('Step 3/4 · Held-out evaluation complete. Saving depot observations.')
                    dep.save_data(name,frame,capacity)
                    st.session_state['nd_candidate_'+dep._slug(name)]=result
                    progress.progress(1.0,text='Step 4/4 · Candidate model and evaluation ready')
                    st.success('Depot model trained and saved as a candidate. Original models are unchanged.')
                except Exception as exc:
                    st.error(f'Training failed: {exc}')
        result=st.session_state.get('nd_candidate_'+dep._slug(name))
        if result:
            st.markdown('#### Model evaluation · held-out historical observations')
            comparison=[{'Model':'New depot LSTM',**result['candidate_metrics']}]
            if result.get('previous_active_metrics'):
                comparison.append({'Model':'Previous active LSTM',**result['previous_active_metrics']})
            st.dataframe(pd.DataFrame(comparison),hide_index=True,use_container_width=True)
            st.caption('The final 15% of readings were excluded from training and validation.')
            try:
                candidate_model, candidate_scalers = dep._load_model(name, result['version'])
                series = dep._regular(frame)
                X_test, y_test, starts_test = dep._windows(series)
                cutoff = int(len(series) * .85)
                selected = starts_test >= cutoff
                pred_test = dep._predict(candidate_model, candidate_scalers, X_test[selected])
                _evaluation_dashboard(y_test[selected],pred_test,capacity,'Depot-specific LSTM',baseline=X_test[selected,0],historical_note='Chronological held-out test: final 15% of the uploaded series was not used for model fitting or validation.')
                times_test = series.index[96:][selected]
                _chart(pd.Series(y_test[selected][-192:], index=times_test[-192:]),
                       pd.Series(pred_test[-192:], index=times_test[-192:]),
                       'Held-out evaluation | Actual vs predicted demand')
            except Exception as exc:
                st.warning(f'Could not display held-out chart: {exc}')
            if st.button('Activate this depot-specific model',type='primary'):
                dep.activate(name,result['version'])
                st.success('New depot model activated. You can forecast below.')
            if dep.active_paths(name)[0].exists() and dep.active_paths(name)[1].exists():
                st.markdown('#### Predict · depot-specific LSTM')
                period=_forecast_period('nd_own',frame)
                forecast_key='nd_own_forecast_'+dep._slug(name)
                if st.button('Generate depot-specific forecast', type='primary',disabled=period is None):
                    try:
                        with st.spinner('Loading depot-specific LSTM'):
                            own_model, own_scalers = dep._load_model(name)
                        _run_forecast(own_model,own_scalers,frame,period,forecast_key)
                    except Exception as exc:
                        st.error(f'Forecast could not be generated: {exc}')
                forecast=_cached_forecast_result(forecast_key,period) if period else None
                if forecast is not None:
                    _forecast_results(forecast,dep._regular(frame),capacity,name,'depot_specific_forecast.csv',period[3])

"""Per-organisation, per-depot snapshot of the latest generated demand forecast.

Results are written only after successful inference. No inference happens when reading.
"""
import json
from html import escape
import os
from pathlib import Path
import pandas as pd


def _path(depot):
    import depot_updates as dep
    return dep.depot_dir(depot) / 'latest_forecast.json'


def save(depot, prediction, capacity, model_label, model_version=None):
    """Persist timestamped predictions; never share results between depots."""
    series = pd.Series(prediction, dtype=float).sort_index()
    if series.empty or series.isna().any():
        raise ValueError('Cannot save empty or invalid forecast.')
    payload = {
        'depot': depot,
        'model': model_label,
        'model_version': str(model_version or 'active'),
        'capacity_kw': float(capacity),
        'generated_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'timestamps': [pd.Timestamp(t).isoformat() for t in series.index],
        'demand_kw': [float(v) for v in series.to_numpy()],
    }
    path = _path(depot)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload), encoding='utf-8')
    os.replace(tmp, path)
    return payload


def load(depot):
    path = _path(depot)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('depot') != depot:
            return None
        if not payload.get('timestamps') or len(payload['timestamps']) != len(payload['demand_kw']):
            return None
        return payload
    except (OSError, ValueError, KeyError, TypeError):
        return None


def render(depot, heading='Latest forecast results'):
    import streamlit as st
    import plotly.graph_objects as go
    payload = load(depot)
    st.subheader(heading)
    if not payload:
        st.info(f'No future demand forecast is saved for {depot}. The historical LSTM model and its evaluation results may still be available.')
        return False
    times = pd.to_datetime(payload['timestamps'])
    series = pd.Series(payload['demand_kw'], index=times)
    capacity = float(payload['capacity_kw'])
    exceed = int((series > capacity).sum())
    above_warning = int((series >= capacity * .8).sum())
    st.caption(f"{depot} · {payload['model']} · version {payload.get('model_version', 'active')} · generated {payload['generated_at_utc'][:16].replace('T',' ')} UTC")
    peak_at = series.idxmax()
    # Local, responsive styling: do not change the Overview page's other widgets.
    forecast_metrics = [
        ('Highest expected demand', f'{series.max():.2f} kW'),
        ('Grid capacity', f'{capacity:.2f} kW'),
        ('Maximum predicted exceedance', f'{max(0., float(series.max())-capacity):.2f} kW'),
        ('Time above capacity', f'{exceed*.25:g} hours'),
    ]
    cards = ''.join(
        '<div class="optivolt-forecast-metric">'
        f'<div class="optivolt-forecast-label">{escape(label)}</div>'
        f'<div class="optivolt-forecast-value">{escape(value)}</div>'
        '</div>'
        for label, value in forecast_metrics
    )
    st.markdown(
        '<style>'
        '.optivolt-forecast-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin:12px 0 18px}'
        '.optivolt-forecast-metric{box-sizing:border-box;min-width:0;min-height:152px;padding:24px 22px;'
        'border-radius:18px;background:linear-gradient(145deg,#11264A,#234A78);'
        'border:1px solid #355D8C;box-shadow:0 10px 25px rgba(17,38,74,.13)}'
        '.optivolt-forecast-label{color:#C5D9F5;font-size:16px;line-height:1.4;overflow-wrap:anywhere}'
        '.optivolt-forecast-value{color:#FFFFFF;font-size:clamp(25px,2.6vw,43px);font-weight:750;'
        'line-height:1.25;margin-top:16px;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}'
        '@media(max-width:1100px){.optivolt-forecast-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}'
        '@media(max-width:550px){.optivolt-forecast-metrics{grid-template-columns:1fr}}'
        '</style><div class="optivolt-forecast-metrics">' + cards + '</div>',
        unsafe_allow_html=True,
    )
    st.caption(f'Forecast peak expected: {peak_at:%d %b %Y at %H:%M} · Forecast window: {times.min():%d %b %Y %H:%M} to {times.max():%d %b %Y %H:%M}')
    if exceed:
        st.warning(f'Predicted demand could exceed the depot grid limit for {exceed * .25:g} hours. '
                   f'At its busiest, the depot may need {max(0., float(series.max()) - capacity):.2f} kW '
                   'more than the configured capacity. Consider staggering vehicle charging, '
                   'reducing peak charging, and checking the configured grid limit. '
                   'This is a forecast, not a measured overload.')
    elif above_warning:
        st.info(f'Predicted demand stays within grid capacity, but reaches at least 80% of '
                f'the limit for {above_warning * .25:g} hours. Consider monitoring busy charging periods.')
    else:
        st.success('Predicted demand stays below 80% of the configured grid capacity during this forecast.')
    st.caption(f'Total predicted electricity use: {series.sum() * .25:.1f} kWh. '
               'These results are saved predictions, not live meter readings.')
    fig = go.Figure()
    fig.add_scatter(x=series.index, y=series.values, mode='lines', name='Predicted demand', line=dict(color='#087F8C', width=3))
    fig.add_hline(y=capacity, line_dash='dash', line_color='#DC2626', annotation_text='Grid capacity')
    fig.add_hline(y=capacity * .8, line_dash='dot', line_color='#D97706', annotation_text='80% warning')
    fig.update_layout(height=330, font=dict(size=15,color='#14243B'), paper_bgcolor='#FFFFFF', plot_bgcolor='#F7FAFF',
                      margin=dict(l=50,r=30,t=30,b=50), xaxis_title='Date and time', yaxis_title='Demand (kW)',
                      xaxis=dict(gridcolor='#E2E8F0'), yaxis=dict(gridcolor='#E2E8F0'), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    table = pd.DataFrame({'Date and time':times, 'Predicted demand (kW)':series.values.round(2),
                          'Grid capacity status':['Exceeded' if v>capacity else 'Within limit' for v in series.values]})
    with st.expander('Forecast values and grid capacity', expanded=False):
        st.dataframe(table, hide_index=True, use_container_width=True, height=370,
                     column_config={'Date and time':st.column_config.DatetimeColumn('Date and time',format='DD MMM YYYY, HH:mm'),
                                    'Predicted demand (kW)':st.column_config.NumberColumn('Predicted demand (kW)',format='%.2f'),
                                    'Grid capacity status':st.column_config.TextColumn('Grid status')})
    st.download_button('Download forecast CSV', table.to_csv(index=False), file_name=f'{depot.replace(" ","_")}_forecast.csv', mime='text/csv')
    return True

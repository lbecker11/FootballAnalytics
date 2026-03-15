import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title='Profitability Analysis', layout='wide')
st.title('Profitability Analysis')
st.markdown('Expected vs actual profit, and forward projection for next N bets.')

if 'records' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

records = st.session_state['records']

STARTING_BANKROLL = 10000
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
model_colors = dict(zip(records.keys(), colors))

overlay = st.checkbox('Overlay all models')

if overlay:
    # --- Overlay cumulative actual profit for all models ---
    st.subheader('Cumulative actual profit — all models')

    fig = go.Figure()
    for name, rdf in records.items():
        if rdf.empty:
            continue
        cumulative = rdf['profit'].cumsum().reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative) + 1)),
            y=cumulative.values,
            name=name,
            line=dict(color=model_colors[name])
        ))
    fig.add_hline(y=0, line_dash='dot', line_color='grey', annotation_text='Break even')
    fig.update_layout(xaxis_title='Bet number', yaxis_title='Cumulative profit (€)',
                      title='All models — cumulative actual profit')
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    summary_rows = []
    for name, rdf in records.items():
        if rdf.empty:
            continue
        summary_rows.append({
            'model': name,
            'bets': len(rdf),
            'won': rdf['won'].sum(),
            'win rate (%)': round(rdf['won'].mean() * 100, 1),
            'total profit (€)': round(rdf['profit'].sum(), 2),
            'avg profit/bet (€)': round(rdf['profit'].mean(), 2),
            'final bankroll (€)': round(rdf['bankroll'].iloc[-1], 2),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.stop()

# --- Single model view ---
model = st.selectbox('Model', list(records.keys()))
records_df = records[model].copy()

if records_df.empty:
    st.warning(f'No bets flagged for {model}.')
    st.stop()

records_df['bet_number'] = range(1, len(records_df) + 1)
records_df['expected_profit'] = records_df['ev'] * records_df['stake']
records_df['cumulative_expected'] = records_df['expected_profit'].cumsum()
records_df['cumulative_actual'] = records_df['profit'].cumsum()

# --- Expected vs actual profit ---
st.subheader('Expected vs actual profit per bet')

fig = go.Figure()
fig.add_trace(go.Scatter(x=records_df['bet_number'], y=records_df['cumulative_expected'],
                         name='Expected profit', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(x=records_df['bet_number'], y=records_df['cumulative_actual'],
                         name='Actual profit', line=dict(color=model_colors[model])))
fig.add_hline(y=0, line_dash='dot', line_color='grey')
fig.update_layout(xaxis_title='Bet number', yaxis_title='Cumulative profit (€)',
                  title='Cumulative expected vs actual profit')
st.plotly_chart(fig, use_container_width=True)

# --- Per bet stats ---
st.subheader('Per bet statistics')

col1, col2, col3 = st.columns(3)
col1.metric('Avg expected profit per bet', f'€{records_df["expected_profit"].mean():.2f}')
col2.metric('Avg actual profit per bet', f'€{records_df["profit"].mean():.2f}')
col3.metric('Std dev profit per bet', f'€{records_df["profit"].std():.2f}')

st.divider()

# --- Forward projection ---
st.subheader('Forward projection')

n_bets = st.slider('Number of future bets to simulate', min_value=10, max_value=500, value=100, step=10)
n_simulations = 1000

mean_profit = records_df['profit'].mean()
std_profit = records_df['profit'].std()

np.random.seed(42)
simulations = np.random.normal(mean_profit, std_profit, size=(n_simulations, n_bets)).cumsum(axis=1)

mean_path = simulations.mean(axis=0)
upper_95 = np.percentile(simulations, 97.5, axis=0)
lower_95 = np.percentile(simulations, 2.5, axis=0)

x = list(range(1, n_bets + 1))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=x + x[::-1],
                          y=upper_95.tolist() + lower_95.tolist()[::-1],
                          fill='toself', fillcolor='rgba(0,100,255,0.1)',
                          line=dict(color='rgba(255,255,255,0)'),
                          name='95% confidence interval'))
fig2.add_trace(go.Scatter(x=x, y=mean_path, name='Expected path',
                          line=dict(color=model_colors[model])))
fig2.add_hline(y=0, line_dash='dash', line_color='grey', annotation_text='Break even')
fig2.update_layout(xaxis_title='Future bet number', yaxis_title='Projected cumulative profit (€)',
                   title=f'Forward projection — next {n_bets} bets ({n_simulations} simulations)')
st.plotly_chart(fig2, use_container_width=True)

prob_profit = (simulations[:, -1] > 0).mean() * 100
col1, col2, col3 = st.columns(3)
col1.metric('Expected total profit', f'€{mean_path[-1]:.2f}')
col2.metric('95% CI', f'€{lower_95[-1]:.0f} to €{upper_95[-1]:.0f}')
col3.metric('Probability of profit', f'{prob_profit:.1f}%')

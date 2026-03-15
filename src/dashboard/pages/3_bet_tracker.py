import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Bet Tracker', layout='wide')
st.title('Bet Tracker')
st.markdown('Per bet result tracking with bankroll development.')

if 'records' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

records = st.session_state['records']
histories = st.session_state['histories']

STARTING_BANKROLL = 10000

model_options = ['All models (overlay)'] + list(records.keys())
model = st.selectbox('Model', model_options)

# --- Bankroll chart ---
st.subheader('Bankroll development')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

if model == 'All models (overlay)':
    fig = go.Figure()
    for (name, color) in zip(records.keys(), colors):
        s = histories[name]
        fig.add_trace(go.Scatter(x=list(range(len(s))), y=s.values,
                                 name=name, line=dict(color=color)))
    fig.add_hline(y=STARTING_BANKROLL, line_dash='dash', line_color='grey',
                  annotation_text='Starting bankroll')
    fig.update_layout(xaxis_title='Bet number', yaxis_title='Bankroll (€)',
                      title='All models — bankroll development')
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

# Single model view
s = histories[model]
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(s))), y=s.values,
                         name=model, line=dict(color=colors[list(records.keys()).index(model)])))
fig.add_hline(y=STARTING_BANKROLL, line_dash='dash', line_color='grey',
              annotation_text='Starting bankroll')
fig.update_layout(xaxis_title='Bet number', yaxis_title='Bankroll (€)')
st.plotly_chart(fig, use_container_width=True)

# --- Bet log ---
st.subheader('Bet log')

bets = records[model].copy()
bets['match'] = bets['home_team'] + ' vs ' + bets['away_team']
bets['result'] = bets['won'].map({True: 'Won', False: 'Lost'})
bets['bankroll before (€)'] = bets['bankroll'] - bets['profit']

tracker_df = bets[[
    'match', 'season_name', 'matchday', 'bet_outcome', 'odds',
    'stake', 'result', 'profit', 'bankroll before (€)', 'bankroll'
]].rename(columns={
    'season_name': 'season',
    'bet_outcome': 'bet',
    'stake': 'stake (€)',
    'profit': 'profit (€)',
    'bankroll': 'bankroll after (€)',
})

def colour_result(val):
    if val == 'Won':
        return 'background-color: #d4edda; color: #155724'
    elif val == 'Lost':
        return 'background-color: #f8d7da; color: #721c24'
    return ''

def colour_profit(val):
    if isinstance(val, float):
        return 'color: green' if val > 0 else 'color: red'
    return ''

st.dataframe(
    tracker_df.style
        .applymap(colour_result, subset=['result'])
        .applymap(colour_profit, subset=['profit (€)'])
        .format({'odds': '{:.2f}', 'stake (€)': '{:.2f}', 'profit (€)': '{:.2f}',
                 'bankroll before (€)': '{:.2f}', 'bankroll after (€)': '{:.2f}'}),
    use_container_width=True,
    height=500
)

won = bets['won'].sum()
lost = (~bets['won']).sum()
st.caption(f'{len(bets)} bets — {won} won, {lost} lost')

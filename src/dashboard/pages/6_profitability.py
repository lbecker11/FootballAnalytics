import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(page_title='Profitability Analysis', layout='wide')
st.title('Profitability Analysis')
st.markdown('Expected vs actual profit, and forward projection for next N bets.')

if 'predictions_df' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

predictions_df = st.session_state['predictions_df']

STARTING_BANKROLL = 10000

model = st.selectbox('Model', ['XGBoost', 'Dixon-Coles'])

if model == 'XGBoost':
    bets = predictions_df[predictions_df['bet_flag'] == 1].copy()
    bets = bets[bets['team_id'] == bets['home_team_id']].copy()
    bet_outcome_col = 'bet_outcome'
    kelly_h_col = 'kelly_h'
    kelly_a_col = 'kelly_a'
    ev_h_col = 'ev_h'
    ev_a_col = 'ev_a'
else:
    bets = predictions_df[predictions_df['dc_bet_flag'] == 1].copy()
    bets = bets[bets['team_id'] == bets['home_team_id']].copy()
    bet_outcome_col = 'dc_bet_outcome'
    kelly_h_col = 'dc_kelly_h'
    kelly_a_col = 'dc_kelly_a'
    ev_h_col = 'dc_ev_h'
    ev_a_col = 'dc_ev_a'

bets = bets.sort_values(['season_name', 'matchday']).reset_index(drop=True)

# Reconstruct per-bet profit
bankroll = STARTING_BANKROLL
records = []

for _, row in bets.iterrows():
    outcome = row[bet_outcome_col]
    if outcome == 'home_win':
        stake = bankroll * row[kelly_h_col]
        odds = row['PSH']
        ev = row[ev_h_col]
        won = row['outcome'] == 'home_win'
    else:
        stake = bankroll * row[kelly_a_col]
        odds = row['PSA']
        ev = row[ev_a_col]
        won = row['outcome'] == 'away_win'

    expected_profit = ev * stake
    actual_profit = stake * (odds - 1) if won else -stake
    bankroll += actual_profit

    records.append({
        'season': row['season_name'],
        'stake': stake,
        'ev': ev,
        'expected_profit': expected_profit,
        'actual_profit': actual_profit,
        'won': won
    })

records_df = pd.DataFrame(records)

# --- Expected vs actual profit ---
st.subheader('Expected vs actual profit per bet')

records_df['bet_number'] = range(1, len(records_df) + 1)
records_df['cumulative_expected'] = records_df['expected_profit'].cumsum()
records_df['cumulative_actual'] = records_df['actual_profit'].cumsum()

fig = go.Figure()
fig.add_trace(go.Scatter(x=records_df['bet_number'], y=records_df['cumulative_expected'],
                         name='Expected profit', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(x=records_df['bet_number'], y=records_df['cumulative_actual'],
                         name='Actual profit', line=dict(color='green')))
fig.add_hline(y=0, line_dash='dot', line_color='grey')
fig.update_layout(xaxis_title='Bet number', yaxis_title='Cumulative profit (€)',
                  title='Cumulative expected vs actual profit')
st.plotly_chart(fig, use_container_width=True)

# --- Per bet stats ---
st.subheader('Per bet statistics')

col1, col2, col3 = st.columns(3)
col1.metric('Avg expected profit per bet', f'€{records_df["expected_profit"].mean():.2f}')
col2.metric('Avg actual profit per bet', f'€{records_df["actual_profit"].mean():.2f}')
col3.metric('Std dev profit per bet', f'€{records_df["actual_profit"].std():.2f}')

st.divider()

# --- Forward projection ---
st.subheader('Forward projection')

n_bets = st.slider('Number of future bets to simulate', min_value=10, max_value=500, value=100, step=10)
n_simulations = 1000

mean_profit = records_df['actual_profit'].mean()
std_profit = records_df['actual_profit'].std()

# Monte Carlo simulation
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
                          line=dict(color='blue')))
fig2.add_hline(y=0, line_dash='dash', line_color='grey', annotation_text='Break even')
fig2.update_layout(xaxis_title='Future bet number', yaxis_title='Projected cumulative profit (€)',
                   title=f'Forward projection — next {n_bets} bets ({n_simulations} simulations)')
st.plotly_chart(fig2, use_container_width=True)

# Summary stats
prob_profit = (simulations[:, -1] > 0).mean() * 100
col1, col2, col3 = st.columns(3)
col1.metric('Expected total profit', f'€{mean_path[-1]:.2f}')
col2.metric('95% CI', f'€{lower_95[-1]:.0f} to €{upper_95[-1]:.0f}')
col3.metric('Probability of profit', f'{prob_profit:.1f}%')

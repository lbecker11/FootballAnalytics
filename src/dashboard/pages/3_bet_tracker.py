import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Bet Tracker', layout='wide')
st.title('Bet Tracker')
st.markdown('Per bet result tracking with bankroll development.')

if 'predictions_df' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

predictions_df = st.session_state['predictions_df']
bankroll_df = st.session_state['bankroll_df']
bankroll_dc_df = st.session_state['bankroll_dc_df']

STARTING_BANKROLL = 10000

model = st.selectbox('Model', ['XGBoost', 'Dixon-Coles'])

if model == 'XGBoost':
    bets = predictions_df[predictions_df['bet_flag'] == 1].copy()
    bets = bets[bets['team_id'] == bets['home_team_id']].copy()
    bets = bets.sort_values(['season_name', 'matchday']).reset_index(drop=True)
    history = bankroll_df.copy()
    bet_outcome_col = 'bet_outcome'
    kelly_h_col = 'kelly_h'
    kelly_a_col = 'kelly_a'
    odds_h_col = 'PSH'
    odds_a_col = 'PSA'
else:
    bets = predictions_df[predictions_df['dc_bet_flag'] == 1].copy()
    bets = bets[bets['team_id'] == bets['home_team_id']].copy()
    bets = bets.sort_values(['season_name', 'matchday']).reset_index(drop=True)
    history = bankroll_dc_df.copy()
    bet_outcome_col = 'dc_bet_outcome'
    kelly_h_col = 'dc_kelly_h'
    kelly_a_col = 'dc_kelly_a'
    odds_h_col = 'PSH'
    odds_a_col = 'PSA'

# Reconstruct bet tracker from bets
tracker = []
bankroll = STARTING_BANKROLL

for _, row in bets.iterrows():
    outcome = row[bet_outcome_col]
    if outcome == 'home_win':
        stake = bankroll * row[kelly_h_col]
        odds = row[odds_h_col]
        won = row['outcome'] == 'home_win'
    else:
        stake = bankroll * row[kelly_a_col]
        odds = row[odds_a_col]
        won = row['outcome'] == 'away_win'

    bankroll_before = bankroll
    if won:
        profit = stake * (odds - 1)
        bankroll += profit
    else:
        profit = -stake
        bankroll -= stake

    tracker.append({
        'match': row['home_team_name'] + ' vs ' + row['away_team_name'],
        'season': row['season_name'],
        'matchday': row['matchday'],
        'bet': outcome,
        'odds': odds,
        'stake (€)': round(stake, 2),
        'result': 'Won' if won else 'Lost',
        'profit (€)': round(profit, 2),
        'returns (€)': round(stake + profit, 2),
        'bankroll before (€)': round(bankroll_before, 2),
        'bankroll after (€)': round(bankroll, 2),
        'change (%)': round((profit / bankroll_before) * 100, 2)
    })

tracker_df = pd.DataFrame(tracker)

# Bankroll chart
st.subheader('Bankroll development')
history_plot = history.copy()
history_plot.index.name = 'bet number'
fig = px.line(history_plot, x=history_plot.index, y='bankroll')
fig.add_hline(y=STARTING_BANKROLL, line_dash='dash', line_color='grey', annotation_text='Starting bankroll')
fig.update_layout(xaxis_title='Bet number', yaxis_title='Bankroll (€)')
st.plotly_chart(fig, use_container_width=True)

# Tracker table
st.subheader('Bet log')

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
        .applymap(colour_profit, subset=['profit (€)', 'change (%)'])
        .format({'odds': '{:.2f}', 'stake (€)': '{:.2f}', 'profit (€)': '{:.2f}',
                 'returns (€)': '{:.2f}', 'bankroll before (€)': '{:.2f}',
                 'bankroll after (€)': '{:.2f}', 'change (%)': '{:.2f}'}),
    use_container_width=True,
    height=500
)

st.caption(f'{len(tracker_df)} bets — {tracker_df["result"].value_counts().get("Won", 0)} won, {tracker_df["result"].value_counts().get("Lost", 0)} lost')

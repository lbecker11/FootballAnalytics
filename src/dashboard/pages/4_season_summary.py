import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title='Season Summary', layout='wide')
st.title('Season Summary')
st.markdown('Financial KPIs, betting statistics, bet breakdown and risk metrics per season.')

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
    bet_outcome_col = 'bet_outcome'
    kelly_h_col = 'kelly_h'
    kelly_a_col = 'kelly_a'
else:
    bets = predictions_df[predictions_df['dc_bet_flag'] == 1].copy()
    bets = bets[bets['team_id'] == bets['home_team_id']].copy()
    bet_outcome_col = 'dc_bet_outcome'
    kelly_h_col = 'dc_kelly_h'
    kelly_a_col = 'dc_kelly_a'

bets = bets.sort_values(['season_name', 'matchday']).reset_index(drop=True)

# Reconstruct per-bet results with running bankroll
bankroll = STARTING_BANKROLL
records = []

for _, row in bets.iterrows():
    outcome = row[bet_outcome_col]
    if outcome == 'home_win':
        stake = bankroll * row[kelly_h_col]
        odds = row['PSH']
        won = row['outcome'] == 'home_win'
    else:
        stake = bankroll * row[kelly_a_col]
        odds = row['PSA']
        won = row['outcome'] == 'away_win'

    profit = stake * (odds - 1) if won else -stake
    bankroll += profit

    records.append({
        'season': row['season_name'],
        'bet': outcome,
        'stake': stake,
        'odds': odds,
        'won': won,
        'profit': profit,
        'bankroll': bankroll
    })

records_df = pd.DataFrame(records)

# --- Overall KPIs ---
st.subheader('Overall financial KPIs')
final_bankroll = records_df['bankroll'].iloc[-1]
total_profit = final_bankroll - STARTING_BANKROLL
total_staked = records_df['stake'].sum()
roi = (total_profit / total_staked) * 100
win_rate = records_df['won'].mean() * 100

# Sharpe ratio — mean profit per bet / std profit per bet
mean_profit = records_df['profit'].mean()
std_profit = records_df['profit'].std()
sharpe = mean_profit / std_profit if std_profit != 0 else 0

# Max drawdown
peak = STARTING_BANKROLL
max_drawdown = 0
for b in records_df['bankroll']:
    if b > peak:
        peak = b
    drawdown = (peak - b) / peak
    max_drawdown = max(max_drawdown, drawdown)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric('Starting bankroll', f'€{STARTING_BANKROLL:,.0f}')
col2.metric('Final bankroll', f'€{final_bankroll:,.2f}')
col3.metric('Total profit', f'€{total_profit:,.2f}')
col4.metric('ROI on stakes', f'{roi:.1f}%')
col5.metric('Win rate', f'{win_rate:.1f}%')
col6.metric('Max drawdown', f'{max_drawdown * 100:.1f}%')

col7, col8 = st.columns(2)
col7.metric('Sharpe ratio', f'{sharpe:.3f}')
col8.metric('Total staked', f'€{total_staked:,.2f}')

st.divider()

# --- Per season summary ---
st.subheader('Per season breakdown')

season_stats = []
for season, group in records_df.groupby('season'):
    s_profit = group['profit'].sum()
    s_staked = group['stake'].sum()
    s_roi = (s_profit / s_staked) * 100 if s_staked > 0 else 0
    s_win_rate = group['won'].mean() * 100
    s_bets = len(group)
    s_won = group['won'].sum()
    s_lost = s_bets - s_won
    home_bets = (group['bet'] == 'home_win').sum()
    away_bets = (group['bet'] == 'away_win').sum()
    season_stats.append({
        'season': season,
        'bets': s_bets,
        'won': s_won,
        'lost': s_lost,
        'win rate (%)': round(s_win_rate, 1),
        'staked (€)': round(s_staked, 2),
        'profit (€)': round(s_profit, 2),
        'ROI (%)': round(s_roi, 1),
        'home bets': home_bets,
        'away bets': away_bets
    })

season_df = pd.DataFrame(season_stats)

def colour_profit(val):
    if isinstance(val, float) or isinstance(val, int):
        return 'color: green' if val > 0 else 'color: red'
    return ''

st.dataframe(
    season_df.style.applymap(colour_profit, subset=['profit (€)', 'ROI (%)']),
    use_container_width=True
)

# Profit per season bar chart
fig = px.bar(season_df, x='season', y='profit (€)', color='profit (€)',
             color_continuous_scale=['red', 'green'],
             title='Profit per season')
fig.add_hline(y=0, line_dash='dash', line_color='grey')
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Risk metrics ---
st.subheader('Risk metrics')

col1, col2, col3 = st.columns(3)
col1.metric('Avg bet size', f'€{records_df["stake"].mean():.2f}')
col2.metric('Largest bet', f'€{records_df["stake"].max():.2f}')
col3.metric('Smallest bet', f'€{records_df["stake"].min():.2f}')

# Winning and losing streaks
streaks = records_df['won'].tolist()
best_streak = worst_streak = cur_win = cur_lose = 0
for w in streaks:
    if w:
        cur_win += 1
        cur_lose = 0
    else:
        cur_lose += 1
        cur_win = 0
    best_streak = max(best_streak, cur_win)
    worst_streak = max(worst_streak, cur_lose)

largest_win = records_df[records_df['profit'] > 0]['profit'].max()
largest_loss = records_df[records_df['profit'] < 0]['profit'].min()

col4, col5, col6, col7 = st.columns(4)
col4.metric('Best winning streak', best_streak)
col5.metric('Worst losing streak', worst_streak)
col6.metric('Largest win', f'€{largest_win:.2f}')
col7.metric('Largest loss', f'€{largest_loss:.2f}')

# Bet breakdown pie chart
bet_counts = records_df['bet'].value_counts().reset_index()
bet_counts.columns = ['outcome', 'count']
fig2 = px.pie(bet_counts, names='outcome', values='count', title='Bet breakdown by outcome')
st.plotly_chart(fig2, use_container_width=True)

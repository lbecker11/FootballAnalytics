import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Season Summary', layout='wide')
st.title('Season Summary')
st.markdown('Financial KPIs, betting statistics and risk metrics per season.')

if 'records' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

records = st.session_state['records']

STARTING_BANKROLL = 10000

model = st.selectbox('Model', list(records.keys()))

bets = records[model].copy()
if bets.empty:
    st.warning(f'No bets flagged for {model}.')
    st.stop()

bets = bets.sort_values(['season_name', 'matchday']).reset_index(drop=True)

# --- Overall KPIs ---
st.subheader('Overall financial KPIs')

final_bankroll = bets['bankroll'].iloc[-1]
total_profit = final_bankroll - STARTING_BANKROLL
total_staked = bets['stake'].sum()
roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
win_rate = bets['won'].mean() * 100

mean_profit = bets['profit'].mean()
std_profit = bets['profit'].std()
sharpe = mean_profit / std_profit if std_profit != 0 else 0

peak = STARTING_BANKROLL
max_drawdown = 0
for b in bets['bankroll']:
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
for season, group in bets.groupby('season_name'):
    s_profit = group['profit'].sum()
    s_staked = group['stake'].sum()
    s_roi = (s_profit / s_staked) * 100 if s_staked > 0 else 0
    season_stats.append({
        'season': season,
        'bets': len(group),
        'won': group['won'].sum(),
        'lost': (~group['won']).sum(),
        'win rate (%)': round(group['won'].mean() * 100, 1),
        'staked (€)': round(s_staked, 2),
        'profit (€)': round(s_profit, 2),
        'ROI (%)': round(s_roi, 1),
        'home bets': (group['bet_outcome'] == 'home_win').sum(),
        'away bets': (group['bet_outcome'] == 'away_win').sum(),
    })

season_df = pd.DataFrame(season_stats)

def colour_profit(val):
    if isinstance(val, (float, int)):
        return 'color: green' if val > 0 else 'color: red'
    return ''

st.dataframe(
    season_df.style.applymap(colour_profit, subset=['profit (€)', 'ROI (%)']),
    use_container_width=True
)

fig = px.bar(season_df, x='season', y='profit (€)', color='profit (€)',
             color_continuous_scale=['red', 'green'],
             title='Profit per season')
fig.add_hline(y=0, line_dash='dash', line_color='grey')
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Risk metrics ---
st.subheader('Risk metrics')

col1, col2, col3 = st.columns(3)
col1.metric('Avg bet size', f'€{bets["stake"].mean():.2f}')
col2.metric('Largest bet', f'€{bets["stake"].max():.2f}')
col3.metric('Smallest bet', f'€{bets["stake"].min():.2f}')

streaks = bets['won'].tolist()
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

wins = bets[bets['profit'] > 0]['profit']
losses = bets[bets['profit'] < 0]['profit']

col4, col5, col6, col7 = st.columns(4)
col4.metric('Best winning streak', best_streak)
col5.metric('Worst losing streak', worst_streak)
col6.metric('Largest win', f'€{wins.max():.2f}' if not wins.empty else 'N/A')
col7.metric('Largest loss', f'€{losses.min():.2f}' if not losses.empty else 'N/A')

bet_counts = bets['bet_outcome'].value_counts().reset_index()
bet_counts.columns = ['outcome', 'count']
fig2 = px.pie(bet_counts, names='outcome', values='count', title='Bet breakdown by direction')
st.plotly_chart(fig2, use_container_width=True)

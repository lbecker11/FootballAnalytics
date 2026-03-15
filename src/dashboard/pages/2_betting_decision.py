import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Betting Decision', layout='wide')
st.title('Betting Decision')
st.markdown('Best bet per match — stake, potential profit/loss and risk assessment.')

if 'predictions_df' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

predictions_df = st.session_state['predictions_df']

# Filter to one row per match (home perspective) with at least one flagged bet
home = predictions_df[predictions_df['team_id'] == predictions_df['home_team_id']].copy()
flagged = home[(home['bet_flag'] == 1) | (home['dc_bet_flag'] == 1)].copy()

# Filters
col1, col2 = st.columns(2)
with col1:
    seasons = ['All'] + sorted(flagged['season_name'].unique().tolist())
    selected_season = st.selectbox('Season', seasons)
with col2:
    model_filter = st.selectbox('Model', ['Both', 'XGBoost', 'Dixon-Coles'])

if selected_season != 'All':
    flagged = flagged[flagged['season_name'] == selected_season]
if model_filter == 'XGBoost':
    flagged = flagged[flagged['bet_flag'] == 1]
elif model_filter == 'Dixon-Coles':
    flagged = flagged[flagged['dc_bet_flag'] == 1]

STARTING_BANKROLL = 10000

flagged['match'] = flagged['home_team_name'] + ' vs ' + flagged['away_team_name']

# Determine best model bet per match — highest EV across XGB and DC
def best_bet(row):
    options = []
    if row['bet_flag'] == 1:
        ev = row['ev_h'] if row['bet_outcome'] == 'home_win' else row['ev_a']
        kelly = row['kelly_h'] if row['bet_outcome'] == 'home_win' else row['kelly_a']
        odds = row['PSH'] if row['bet_outcome'] == 'home_win' else row['PSA']
        options.append(('XGBoost', row['bet_outcome'], ev, kelly, odds))
    if row['dc_bet_flag'] == 1:
        ev = row['dc_ev_h'] if row['dc_bet_outcome'] == 'home_win' else row['dc_ev_a']
        kelly = row['dc_kelly_h'] if row['dc_bet_outcome'] == 'home_win' else row['dc_kelly_a']
        odds = row['PSH'] if row['dc_bet_outcome'] == 'home_win' else row['PSA']
        options.append(('Dixon-Coles', row['dc_bet_outcome'], ev, kelly, odds))
    if not options:
        return pd.Series({'best_model': None, 'best_outcome': None, 'best_ev': None, 'best_kelly': None, 'best_odds': None})
    best = max(options, key=lambda x: x[2])
    return pd.Series({'best_model': best[0], 'best_outcome': best[1], 'best_ev': best[2], 'best_kelly': best[3], 'best_odds': best[4]})

flagged[['best_model', 'best_outcome', 'best_ev', 'best_kelly', 'best_odds']] = flagged.apply(best_bet, axis=1)

flagged['stake_eur'] = (flagged['best_kelly'] * STARTING_BANKROLL).round(2)
flagged['potential_profit'] = (flagged['stake_eur'] * (flagged['best_odds'] - 1)).round(2)
flagged['potential_loss'] = flagged['stake_eur']

def confidence(ev):
    if ev >= 0.5:
        return 'High'
    elif ev >= 0.25:
        return 'Medium'
    else:
        return 'Low'

flagged['confidence'] = flagged['best_ev'].apply(confidence)

def colour_confidence(val):
    colours = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    return f'color: {colours.get(val, "black")}'

flagged['result'] = flagged.apply(
    lambda r: 'Won' if r['best_outcome'] == r['outcome'] else 'Lost', axis=1
)

display = flagged[[
    'match', 'season_name', 'matchday',
    'best_model', 'best_outcome', 'best_odds',
    'best_ev', 'best_kelly',
    'stake_eur', 'potential_profit', 'potential_loss',
    'confidence', 'outcome', 'result'
]].rename(columns={
    'season_name': 'season',
    'best_model': 'model',
    'best_outcome': 'bet',
    'best_odds': 'odds',
    'best_ev': 'ev',
    'best_kelly': 'kelly',
    'stake_eur': 'stake (€)',
    'potential_profit': 'profit if win (€)',
    'potential_loss': 'loss if lose (€)',
    'outcome': 'actual'
})

def colour_result(val):
    if val == 'Won':
        return 'background-color: #d4edda; color: #155724'
    elif val == 'Lost':
        return 'background-color: #f8d7da; color: #721c24'
    return ''

st.dataframe(
    display.style
        .applymap(colour_confidence, subset=['confidence'])
        .applymap(colour_result, subset=['result'])
        .format({'ev': '{:.3f}', 'kelly': '{:.3f}', 'odds': '{:.2f}',
                 'stake (€)': '{:.2f}', 'profit if win (€)': '{:.2f}', 'loss if lose (€)': '{:.2f}'}),
    use_container_width=True,
    height=500
)

st.caption(f'Showing {len(display)} flagged matches')

# Confidence breakdown chart
st.subheader('Confidence breakdown')
conf_counts = flagged['confidence'].value_counts().reset_index()
conf_counts.columns = ['confidence', 'count']
fig = px.bar(conf_counts, x='confidence', y='count', color='confidence',
             color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'})
st.plotly_chart(fig, use_container_width=True)

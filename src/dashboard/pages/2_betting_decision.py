import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Betting Decision', layout='wide')
st.title('Betting Decision')
st.markdown('Flagged bets per model — stake, potential profit/loss and outcome.')

if 'records' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

records = st.session_state['records']

STARTING_BANKROLL = 10000

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    model = st.selectbox('Model', list(records.keys()))
with col2:
    seasons = ['All'] + sorted(records[model]['season_name'].unique().tolist())
    selected_season = st.selectbox('Season', seasons)
with col3:
    selected_outcome = st.selectbox('Bet direction', ['All', 'home_win', 'away_win'])

bets = records[model].copy()
if selected_season != 'All':
    bets = bets[bets['season_name'] == selected_season]
if selected_outcome != 'All':
    bets = bets[bets['bet_outcome'] == selected_outcome]

if bets.empty:
    st.warning('No bets match the selected filters.')
    st.stop()

bets['match'] = bets['home_team'] + ' vs ' + bets['away_team']
bets['stake_eur'] = bets['stake'].round(2)
bets['potential_profit'] = (bets['stake'] * (bets['odds'] - 1)).round(2)

def confidence(ev):
    if ev >= 0.5:
        return 'High'
    elif ev >= 0.25:
        return 'Medium'
    return 'Low'

bets['confidence'] = bets['ev'].apply(confidence)
bets['result'] = bets['won'].map({True: 'Won', False: 'Lost'})

display = bets[[
    'match', 'season_name', 'matchday',
    'bet_outcome', 'odds', 'ev', 'stake_eur', 'potential_profit', 'stake_eur',
    'confidence', 'actual_outcome', 'result'
]].rename(columns={
    'season_name': 'season',
    'bet_outcome': 'bet',
    'ev': 'EV',
    'stake_eur': 'stake (€)',
    'potential_profit': 'profit if win (€)',
    'actual_outcome': 'actual',
})
# drop duplicate stake col from rename
display = display.loc[:, ~display.columns.duplicated()]

def colour_confidence(val):
    return {'High': 'color: green', 'Medium': 'color: orange', 'Low': 'color: red'}.get(val, '')

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
        .format({'EV': '{:.3f}', 'odds': '{:.2f}',
                 'stake (€)': '{:.2f}', 'profit if win (€)': '{:.2f}'}),
    use_container_width=True,
    height=500
)

st.caption(f'Showing {len(display)} flagged bets — {bets["won"].sum()} won, {(~bets["won"]).sum()} lost')

# Confidence breakdown
st.subheader('Confidence breakdown')
conf_counts = bets['confidence'].value_counts().reset_index()
conf_counts.columns = ['confidence', 'count']
fig = px.bar(conf_counts, x='confidence', y='count', color='confidence',
             color_discrete_map={'High': 'green', 'Medium': 'orange', 'Low': 'red'})
st.plotly_chart(fig, use_container_width=True)

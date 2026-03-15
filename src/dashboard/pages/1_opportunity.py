import streamlit as st
import pandas as pd

st.set_page_config(page_title='Opportunity Analysis', layout='wide')
st.title('Opportunity Analysis')
st.markdown('Per match breakdown of probabilities, edge, EV and Kelly stake for both models.')

if 'predictions_df' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

predictions_df = st.session_state['predictions_df']

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    seasons = ['All'] + sorted(predictions_df['season_name'].unique().tolist())
    selected_season = st.selectbox('Season', seasons)
with col2:
    selected_outcome = st.selectbox('Bet outcome', ['All', 'home_win', 'away_win'])
with col3:
    min_ev = st.slider('Minimum EV (XGBoost)', min_value=-1.0, max_value=2.0, value=0.0, step=0.05)

# Filter data
filtered = predictions_df.copy()
if selected_season != 'All':
    filtered = filtered[filtered['season_name'] == selected_season]
if selected_outcome != 'All':
    filtered = filtered[filtered['bet_outcome'] == selected_outcome]
filtered = filtered[filtered['ev_h'].fillna(0) >= min_ev]

# Build display table
home_mask = filtered['team_id'] == filtered['home_team_id']
display = filtered[home_mask].copy()

display['match'] = display['home_team_name'] + ' vs ' + display['away_team_name']

cols = [
    'match', 'season_name', 'matchday',
    'PSH', 'PSD', 'PSA',
    'fair_h', 'fair_d', 'fair_a',
    'pred_home_win', 'pred_draw', 'pred_away_win',
    'dc_home_win', 'dc_draw', 'dc_away_win',
    'edge_h', 'edge_d', 'edge_a',
    'dc_edge_h', 'dc_edge_d', 'dc_edge_a',
    'ev_h', 'ev_d', 'ev_a',
    'dc_ev_h', 'dc_ev_a',
    'kelly_h', 'kelly_a',
    'dc_kelly_h', 'dc_kelly_a',
    'bet_outcome', 'dc_bet_outcome',
    'outcome'
]

display = display[cols].rename(columns={
    'season_name': 'season',
    'pred_home_win': 'xgb_h',
    'pred_draw': 'xgb_d',
    'pred_away_win': 'xgb_a',
    'bet_outcome': 'xgb_bet',
    'dc_bet_outcome': 'dc_bet',
    'outcome': 'actual'
})

# Colour positive edge green, negative red
def colour_edge(val):
    if isinstance(val, float):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'
    return ''

edge_cols = ['edge_h', 'edge_d', 'edge_a', 'dc_edge_h', 'dc_edge_d', 'dc_edge_a']

st.dataframe(
    display.style.applymap(colour_edge, subset=[c for c in edge_cols if c in display.columns])
           .format({c: '{:.3f}' for c in display.select_dtypes('float').columns}),
    use_container_width=True,
    height=600
)

st.caption(f'Showing {len(display)} matches')

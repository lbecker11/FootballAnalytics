import streamlit as st
import pandas as pd

st.set_page_config(page_title='Opportunity Analysis', layout='wide')
st.title('Opportunity Analysis')
st.markdown('Per match breakdown of probabilities, edge, EV and Kelly stake for all models.')

if 'predictions_df' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

df = st.session_state['predictions_df']

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    seasons = ['All'] + sorted(df['season_name'].unique().tolist())
    selected_season = st.selectbox('Season', seasons)
with col2:
    selected_outcome = st.selectbox('Bet outcome (XGBoost Classifier)', ['All', 'home_win', 'away_win'])
with col3:
    min_ev = st.slider('Minimum EV (XGBoost Classifier)', min_value=-1.0, max_value=2.0, value=0.0, step=0.05)

filtered = df.copy()
if selected_season != 'All':
    filtered = filtered[filtered['season_name'] == selected_season]
if selected_outcome != 'All':
    filtered = filtered[filtered['bet_outcome'] == selected_outcome]
filtered = filtered[filtered['ev_h'].fillna(0) >= min_ev]

# Compute edge for the three models that don't have dedicated calc_ functions
for prefix, h_col, a_col in [
    ('xgb', 'xgb_home_win', 'xgb_away_win'),
    ('bp',  'bp_home_win',  'bp_away_win'),
    ('stack', 'stack_home_win', 'stack_away_win'),
]:
    filtered[f'{prefix}_edge_h'] = filtered[h_col] - filtered['fair_h']
    filtered[f'{prefix}_edge_a'] = filtered[a_col]  - filtered['fair_a']

filtered['match'] = filtered['home_team_name'] + ' vs ' + filtered['away_team_name']

cols = [
    'match', 'season_name', 'matchday', 'outcome',
    'PSH', 'PSD', 'PSA', 'fair_h', 'fair_d', 'fair_a',
    # XGBoost Classifier
    'pred_home_win', 'pred_draw', 'pred_away_win',
    'edge_h', 'edge_a', 'ev_h', 'ev_a', 'kelly_h', 'kelly_a', 'bet_outcome',
    # Dixon-Coles
    'dc_home_win', 'dc_draw', 'dc_away_win',
    'dc_edge_h', 'dc_edge_a', 'dc_ev_h', 'dc_ev_a', 'dc_kelly_h', 'dc_kelly_a', 'dc_bet_outcome',
    # XGBoost xG
    'xgb_home_win', 'xgb_draw', 'xgb_away_win', 'xgb_edge_h', 'xgb_edge_a',
    # Bivariate Poisson
    'bp_home_win', 'bp_draw', 'bp_away_win', 'bp_edge_h', 'bp_edge_a',
    # Stacking
    'stack_home_win', 'stack_draw', 'stack_away_win', 'stack_edge_h', 'stack_edge_a',
]
# Only keep cols that exist
cols = [c for c in cols if c in filtered.columns]

display = filtered[cols].rename(columns={
    'season_name': 'season',
    'pred_home_win': 'cls_h', 'pred_draw': 'cls_d', 'pred_away_win': 'cls_a',
    'edge_h': 'cls_edge_h', 'edge_a': 'cls_edge_a',
    'ev_h': 'cls_ev_h', 'ev_a': 'cls_ev_a',
    'kelly_h': 'cls_kelly_h', 'kelly_a': 'cls_kelly_a',
    'bet_outcome': 'cls_bet',
    'dc_bet_outcome': 'dc_bet',
    'outcome': 'actual',
})

edge_cols = [c for c in display.columns if 'edge' in c]

def colour_edge(val):
    if isinstance(val, float):
        return 'color: green' if val > 0 else 'color: red'
    return ''

st.dataframe(
    display.style
           .applymap(colour_edge, subset=edge_cols)
           .format({c: '{:.3f}' for c in display.select_dtypes('float').columns}),
    use_container_width=True,
    height=600
)

st.caption(f'Showing {len(display)} matches')

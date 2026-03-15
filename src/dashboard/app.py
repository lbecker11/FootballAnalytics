import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from betting import (
    calc_implied_prob, calc_edge, calc_ev, flag_bets, calc_kelly,
    calc_edge_dc, calc_ev_dc, flag_bets_dc, calc_kelly_dc,
    run_backtest, MODELS,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', '..', 'data')

st.set_page_config(page_title='Football Analytics', layout='wide')


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'all_predictions.csv'), sep=';')

    # Drop rows missing Pinnacle odds
    df = df.dropna(subset=['PSH', 'PSD', 'PSA']).copy()

    # Run backtest for all 5 models (uses actual_outcome as-is from CSV)
    histories = {}
    records = {}
    for name, h_col, d_col, a_col in MODELS:
        sub = df.dropna(subset=[h_col, d_col, a_col])
        histories[name], records[name] = run_backtest(sub, name, h_col, d_col, a_col)

    # Rename to match column names expected by dashboard pages
    df = df.rename(columns={
        'actual_outcome': 'outcome',
        'home_team': 'home_team_name',
        'away_team': 'away_team_name',
    })

    # Pages filter to home rows using team_id == home_team_id
    # all_predictions.csv is already one row per match, so sentinel values work fine
    df['team_id'] = 1
    df['home_team_id'] = 1
    df['away_team_id'] = 0

    # Vig removal + edge/EV/Kelly for XGBoost Classifier and Dixon-Coles
    df = (df
          .pipe(calc_implied_prob)
          .pipe(calc_edge)
          .pipe(calc_ev)
          .pipe(flag_bets)
          .pipe(calc_kelly)
          .pipe(calc_edge_dc)
          .pipe(calc_ev_dc)
          .pipe(flag_bets_dc)
          .pipe(calc_kelly_dc)
    )

    # Build bankroll dfs for pages that reference session_state bankroll_df/bankroll_dc_df
    bankroll_df = pd.DataFrame({'bankroll': histories['XGBoost Classifier'].values})
    bankroll_dc_df = pd.DataFrame({'bankroll': histories['Dixon-Coles'].values})

    return df, histories, records, bankroll_df, bankroll_dc_df


st.title('Football Analytics Dashboard')
st.markdown('Bundesliga betting edge analysis — 5 model comparison')

with st.spinner('Loading data...'):
    predictions_df, histories, records, bankroll_df, bankroll_dc_df = load_data()

st.session_state['predictions_df'] = predictions_df
st.session_state['histories'] = histories
st.session_state['records'] = records
st.session_state['bankroll_df'] = bankroll_df
st.session_state['bankroll_dc_df'] = bankroll_dc_df

st.success(f'Loaded {len(predictions_df)} matches across {predictions_df["season_name"].nunique()} seasons')

# --- Metrics for all 5 models ---
STARTING_BANKROLL = 10000
cols = st.columns(len(MODELS))
for col, (name, *_) in zip(cols, MODELS):
    series = histories[name]
    final = series.iloc[-1]
    ret = (final - STARTING_BANKROLL) / STARTING_BANKROLL * 100
    n_bets = len(series) - 1
    col.metric(name, f'{ret:+.1f}%', f'{n_bets} bets')

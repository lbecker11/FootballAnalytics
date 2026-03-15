import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from betting import (
    rolling_train, eval_rolling, add_dc_probabilities, merge_odds,
    calc_implied_prob, calc_edge, calc_ev, flag_bets, calc_kelly,
    calc_edge_dc, calc_ev_dc, flag_bets_dc, calc_kelly_dc,
    simulate_bankroll, simulate_bankroll_dc,
    df, df_dc, df_odds, BEST_PARAMS, seasons
)

st.set_page_config(page_title='Football Analytics', layout='wide')

@st.cache_data
def load_data():
    predictions_df, le = rolling_train(df, seasons, BEST_PARAMS)
    predictions_df = add_dc_probabilities(predictions_df, df_dc)
    predictions_df = merge_odds(predictions_df, df_odds, df)
    predictions_df = (predictions_df
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
    bankroll_df = simulate_bankroll(predictions_df)
    bankroll_dc_df = simulate_bankroll_dc(predictions_df)
    return predictions_df, le, bankroll_df, bankroll_dc_df

st.title('Football Analytics Dashboard')
st.markdown('Bundesliga betting edge analysis — XGBoost vs Dixon-Coles')

with st.spinner('Running models...'):
    predictions_df, le, bankroll_df, bankroll_dc_df = load_data()

st.session_state['predictions_df'] = predictions_df
st.session_state['le'] = le
st.session_state['bankroll_df'] = bankroll_df
st.session_state['bankroll_dc_df'] = bankroll_dc_df

st.success(f'Loaded {len(predictions_df)} matches across {predictions_df["season_name"].nunique()} seasons')

col1, col2, col3, col4 = st.columns(4)
col1.metric('XGBoost Bets', predictions_df['bet_flag'].sum())
col2.metric('DC Bets', predictions_df['dc_bet_flag'].sum())
col3.metric('XGBoost Return', f'{((bankroll_df["bankroll"].iloc[-1] - 10000) / 10000 * 100):.1f}%')
col4.metric('DC Return', f'{((bankroll_dc_df["bankroll"].iloc[-1] - 10000) / 10000 * 100):.1f}%')

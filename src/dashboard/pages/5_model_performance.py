import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title='Model Performance', layout='wide')
st.title('Model Performance Comparison')
st.markdown('XGBoost vs Dixon-Coles — prediction accuracy, edge, calibration.')

if 'predictions_df' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

predictions_df = st.session_state['predictions_df']

# Filter to home rows only — one row per match
home = predictions_df[predictions_df['team_id'] == predictions_df['home_team_id']].copy()

# --- Classification metrics ---
st.subheader('Classification report')

col1, col2 = st.columns(2)

def get_report(df, pred_col, actual_col):
    predicted = df[[f'{pred_col}_away_win', f'{pred_col}_draw', f'{pred_col}_home_win']].apply(
        lambda r: ['away_win', 'draw', 'home_win'][r.argmax()], axis=1
    )
    report = classification_report(df[actual_col], predicted, output_dict=True)
    return pd.DataFrame(report).T.round(3), predicted

with col1:
    st.markdown('**XGBoost**')
    xgb_report, xgb_preds = get_report(home, 'pred', 'outcome')
    st.dataframe(xgb_report, use_container_width=True)

with col2:
    st.markdown('**Dixon-Coles**')
    dc_report, dc_preds = get_report(home, 'dc', 'outcome')
    st.dataframe(dc_report, use_container_width=True)

st.divider()

# --- Confusion matrices ---
st.subheader('Confusion matrices')

labels = ['away_win', 'draw', 'home_win']

col1, col2 = st.columns(2)

def plot_confusion(actual, predicted, title):
    cm = confusion_matrix(actual, predicted, labels=labels)
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                    color_continuous_scale='Blues',
                    labels={'x': 'Predicted', 'y': 'Actual'},
                    title=title)
    return fig

with col1:
    st.plotly_chart(plot_confusion(home['outcome'], xgb_preds, 'XGBoost'), use_container_width=True)

with col2:
    st.plotly_chart(plot_confusion(home['outcome'], dc_preds, 'Dixon-Coles'), use_container_width=True)

st.divider()

# --- Edge comparison ---
st.subheader('Edge distribution — XGBoost vs Dixon-Coles')

edge_data = pd.DataFrame({
    'edge': pd.concat([home['edge_h'], home['dc_edge_h'],
                       home['edge_a'], home['dc_edge_a']]),
    'model': (['XGBoost'] * len(home) + ['Dixon-Coles'] * len(home)) * 2,
    'outcome': (['home_win'] * len(home) * 2 + ['away_win'] * len(home) * 2)
}).reset_index(drop=True)

fig = px.box(edge_data, x='outcome', y='edge', color='model',
             title='Edge distribution by outcome and model')
fig.add_hline(y=0, line_dash='dash', line_color='grey')
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Positive EV bets comparison ---
st.subheader('Model comparison summary')

summary = pd.DataFrame({
    'metric': [
        'Total bets flagged',
        'Win rate (%)',
        'Avg edge (home)',
        'Avg edge (away)',
        'Avg EV (home)',
        'Avg EV (away)',
    ],
    'XGBoost': [
        predictions_df['bet_flag'].sum(),
        round((home[home['bet_flag'] == 1]['bet_outcome'] == home[home['bet_flag'] == 1]['outcome']).mean() * 100, 1),
        round(home['edge_h'].mean(), 3),
        round(home['edge_a'].mean(), 3),
        round(home['ev_h'].mean(), 3),
        round(home['ev_a'].mean(), 3),
    ],
    'Dixon-Coles': [
        predictions_df['dc_bet_flag'].sum(),
        round((home[home['dc_bet_flag'] == 1]['dc_bet_outcome'] == home[home['dc_bet_flag'] == 1]['outcome']).mean() * 100, 1),
        round(home['dc_edge_h'].mean(), 3),
        round(home['dc_edge_a'].mean(), 3),
        round(home['dc_ev_h'].mean(), 3),
        round(home['dc_ev_a'].mean(), 3),
    ]
})

st.dataframe(summary, use_container_width=True, hide_index=True)

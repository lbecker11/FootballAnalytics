import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title='Model Performance', layout='wide')
st.title('Model Performance Comparison')
st.markdown('All 5 models — prediction accuracy, calibration, edge distribution.')

if 'predictions_df' not in st.session_state:
    st.warning('Please navigate to the home page first to load the data.')
    st.stop()

predictions_df = st.session_state['predictions_df']

MODELS = [
    ('XGBoost Classifier', 'pred',  'pred_home_win',  'pred_draw',  'pred_away_win'),
    ('Dixon-Coles',        'dc',    'dc_home_win',    'dc_draw',    'dc_away_win'),
    ('XGBoost xG',         'xgb',   'xgb_home_win',   'xgb_draw',   'xgb_away_win'),
    ('Bivariate Poisson',  'bp',    'bp_home_win',    'bp_draw',    'bp_away_win'),
    ('Stacking',           'stack', 'stack_home_win', 'stack_draw', 'stack_away_win'),
]

labels = ['away_win', 'draw', 'home_win']


def get_predicted(df, h_col, d_col, a_col):
    return df[[a_col, d_col, h_col]].apply(
        lambda r: labels[r.argmax()], axis=1
    )


# --- Classification summary table ---
st.subheader('Classification summary — all models')

rows = []
for name, prefix, h_col, d_col, a_col in MODELS:
    sub = predictions_df.dropna(subset=[h_col])
    predicted = get_predicted(sub, h_col, d_col, a_col)
    report = classification_report(sub['outcome'], predicted, output_dict=True, zero_division=0)
    rows.append({
        'model': name,
        'accuracy': round(report['accuracy'], 3),
        'home_win recall': round(report.get('home_win', {}).get('recall', 0), 3),
        'draw recall': round(report.get('draw', {}).get('recall', 0), 3),
        'away_win recall': round(report.get('away_win', {}).get('recall', 0), 3),
        'macro F1': round(report['macro avg']['f1-score'], 3),
    })

summary_df = pd.DataFrame(rows)

def colour_metric(val):
    if isinstance(val, float):
        if val >= 0.50:
            return 'color: green'
        elif val >= 0.40:
            return 'color: orange'
        return 'color: red'
    return ''

st.dataframe(
    summary_df.style.applymap(colour_metric,
        subset=['accuracy', 'home_win recall', 'draw recall', 'away_win recall', 'macro F1']),
    use_container_width=True,
    hide_index=True
)

st.divider()

# --- Per-model confusion matrix ---
st.subheader('Confusion matrix')

selected_model = st.selectbox('Model', [name for name, *_ in MODELS])
_, prefix, h_col, d_col, a_col = next(m for m in MODELS if m[0] == selected_model)

sub = predictions_df.dropna(subset=[h_col])
predicted = get_predicted(sub, h_col, d_col, a_col)
cm = confusion_matrix(sub['outcome'], predicted, labels=labels)

fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                color_continuous_scale='Blues',
                labels={'x': 'Predicted', 'y': 'Actual'},
                title=f'{selected_model} — Confusion Matrix')
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Edge distribution — all 5 models ---
st.subheader('Edge distribution — all models')

edge_rows = []
for name, prefix, h_col, d_col, a_col in MODELS:
    sub = predictions_df.dropna(subset=[h_col, 'fair_h', 'fair_a'])
    edge_h = sub[h_col] - sub['fair_h']
    edge_a = sub[a_col] - sub['fair_a']
    for val in edge_h:
        edge_rows.append({'model': name, 'direction': 'home', 'edge': val})
    for val in edge_a:
        edge_rows.append({'model': name, 'direction': 'away', 'edge': val})

edge_df = pd.DataFrame(edge_rows)

direction = st.radio('Direction', ['home', 'away', 'both'], horizontal=True)
if direction != 'both':
    edge_df = edge_df[edge_df['direction'] == direction]

fig2 = px.box(edge_df, x='model', y='edge', color='model',
              title=f'Edge distribution ({direction})')
fig2.add_hline(y=0, line_dash='dash', line_color='grey')
fig2.update_layout(showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

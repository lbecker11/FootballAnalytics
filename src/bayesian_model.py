import pandas as pd
import numpy as np
import os
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
VIS_DIR = os.path.join(SRC_DIR, '..', 'visualisations')

df = pd.read_csv(os.path.join(DATA_DIR, 'bundesliga_features.csv'))

SEASONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']

# DAG structure — (parent, child) edge list
# Each edge means parent influences child
EDGES = [
    ('elo_strength', 'attack_strength'),
    ('opponent_elo_strength', 'defence_strength'),
    ('form', 'attack_strength'),
    ('h2h', 'attack_strength'),
    ('homeaway', 'attack_strength'),
    ('cumulative_points', 'defence_strength'),
    ('attack_strength', 'outcome'),
    ('defence_strength', 'outcome'),
]


def get_actual_outcome(row):
    if row['home_score'] > row['away_score']:
        return 'home_win'
    elif row['home_score'] == row['away_score']:
        return 'draw'
    else:
        return 'away_win'


def bin_features(df):
    # Convert continuous features into low/medium/high categories
    # Bayesian Networks with pgmpy require discrete variables
    df = df.copy()

    df['elo_strength'] = pd.qcut(df['elo_before'], q=3, labels=['low', 'medium', 'high'])
    df['opponent_elo_strength'] = pd.qcut(df['opponent_elo_before'], q=3, labels=['low', 'medium', 'high'])
    df['form'] = pd.qcut(df['rolling_avg_goals'].fillna(df['rolling_avg_goals'].median()),
                         q=3, labels=['low', 'medium', 'high'])
    df['h2h'] = pd.qcut(df['h2h_points'].fillna(0).clip(lower=0) + 0.001,
                        q=3, labels=['low', 'medium', 'high'], duplicates='drop')
    df['homeaway'] = df['homeaway'].map({'Home': 'home', 'Away': 'away'})
    # Bin actual goals into low/medium/high for the network nodes
    df['home_goals'] = pd.cut(df['home_score'], bins=[-1, 0, 2, 20],
                               labels=['zero', 'low', 'high'])
    df['away_goals'] = pd.cut(df['away_score'], bins=[-1, 0, 2, 20],
                               labels=['zero', 'low', 'high'])

    df['outcome'] = df.apply(get_actual_outcome, axis=1)

    # Intermediate nodes — compute from raw numeric columns before binning cumulative_points
    df['attack_strength'] = pd.qcut(
        df['elo_before'].fillna(df['elo_before'].median()) +
        df['rolling_avg_goals'].fillna(0) * 10,
        q=3, labels=['low', 'medium', 'high']
    )
    df['defence_strength'] = pd.qcut(
        df['opponent_elo_before'].fillna(df['opponent_elo_before'].median()) +
        df['cumulative_points'].fillna(0),
        q=3, labels=['low', 'medium', 'high']
    )

    # Now bin cumulative_points for use as a network node
    df['cumulative_points'] = pd.qcut(df['cumulative_points'].fillna(0),
                                      q=3, labels=['low', 'medium', 'high'], duplicates='drop')

    return df


def prepare_network_data(df):
    # Keep only the columns the network needs
    cols = ['elo_strength', 'opponent_elo_strength', 'form', 'h2h',
            'homeaway', 'cumulative_points', 'attack_strength', 'defence_strength',
            'outcome', 'season_name', 'match_id',
            'home_score', 'away_score', 'team_id', 'home_team_id']
    return df[cols].dropna()


def build_and_fit_model(train_df):
    model = DiscreteBayesianNetwork(EDGES)
    network_cols = ['elo_strength', 'opponent_elo_strength', 'form', 'h2h',
                    'homeaway', 'cumulative_points', 'attack_strength', 'defence_strength',
                    'outcome']
    # Bayesian estimator with Dirichlet prior — equivalent_sample_size controls
    # how much the prior smooths the CPTs (higher = more smoothing, useful for small data)
    model.fit(train_df[network_cols],
              estimator=BayesianEstimator,
              prior_type='dirichlet',
              pseudo_counts=2)
    return model


def predict_match(model, inference, row):
    # Build evidence dict from observed features
    evidence = {
        'elo_strength': str(row['elo_strength']),
        'opponent_elo_strength': str(row['opponent_elo_strength']),
        'form': str(row['form']),
        'h2h': str(row['h2h']),
        'homeaway': str(row['homeaway']),
        'cumulative_points': str(row['cumulative_points']),
    }
    try:
        result = inference.query(variables=['outcome'], evidence=evidence, show_progress=False)
        probs = result.values
        states = result.state_names['outcome']
        prob_dict = dict(zip(states, probs))
        return (
            prob_dict.get('home_win', 0),
            prob_dict.get('draw', 0),
            prob_dict.get('away_win', 0)
        )
    except Exception:
        return (1/3, 1/3, 1/3)  # fallback to uniform if inference fails


def rolling_train(df, seasons):
    all_predictions = []

    binned_df = bin_features(df)
    network_df = prepare_network_data(binned_df)

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        predict_season = seasons[i]

        train_df = network_df[network_df['season_name'].isin(train_seasons)]
        predict_df = network_df[network_df['season_name'] == predict_season]

        # Filter to home rows only — one row per match
        predict_home = predict_df[predict_df['team_id'] == predict_df['home_team_id']].copy()

        model = build_and_fit_model(train_df)
        inference = VariableElimination(model)

        home_wins, draws, away_wins = [], [], []
        for _, row in predict_home.iterrows():
            h, d, a = predict_match(model, inference, row)
            home_wins.append(h)
            draws.append(d)
            away_wins.append(a)

        predict_home = predict_home.copy()
        predict_home['bn_home_win'] = home_wins
        predict_home['bn_draw'] = draws
        predict_home['bn_away_win'] = away_wins

        all_predictions.append(predict_home)
        print(f'Fold {i}: trained on {train_seasons}, predicted {predict_season} — {len(predict_home)} matches')

    return pd.concat(all_predictions, ignore_index=True)


def eval_model(predictions_df):
    print('\n--- Bayesian Network Outcome Prediction ---')

    predictions_df['predicted_outcome'] = predictions_df[['bn_home_win', 'bn_draw', 'bn_away_win']].apply(
        lambda r: ['home_win', 'draw', 'away_win'][r.argmax()], axis=1
    )

    labels = ['away_win', 'draw', 'home_win']
    print(classification_report(predictions_df['outcome'], predictions_df['predicted_outcome'], labels=labels))

    cm = confusion_matrix(predictions_df['outcome'], predictions_df['predicted_outcome'], labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    plt.title('Bayesian Network — Outcome Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'bn_confusion_matrix.png'))
    plt.close()

    return predictions_df


def save_predictions(predictions_df):
    cols = ['match_id', 'season_name', 'outcome', 'predicted_outcome',
            'bn_home_win', 'bn_draw', 'bn_away_win']
    predictions_df[cols].to_csv(os.path.join(DATA_DIR, 'bn_predictions.csv'), sep=';', index=False)
    print(f'Saved bn_predictions.csv — {len(predictions_df)} matches')


if __name__ == '__main__':
    predictions_df = rolling_train(df, SEASONS)
    predictions_df = eval_model(predictions_df)
    save_predictions(predictions_df)

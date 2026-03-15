import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
VIS_DIR = os.path.join(SRC_DIR, '..', 'visualisations')

SEASONS = ['2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']


def load_predictions():
    # XGBoost classifier predictions — from betting.py rolling train
    # Contains match_id, pred_home_win, pred_draw, pred_away_win, outcome
    xgb = pd.read_csv(os.path.join(DATA_DIR, 'bundesliga_features.csv'))

    # DC probabilities from bundesliga_features — covers all seasons via rolling DC model
    dc = pd.read_csv(os.path.join(DATA_DIR, 'bundesliga_features.csv'))

    # XGBoost xG regressor predictions
    xg = pd.read_csv(os.path.join(DATA_DIR, 'xg_predictions.csv'), sep=';')

    # Bivariate Poisson predictions
    bp = pd.read_csv(os.path.join(DATA_DIR, 'bp_predictions.csv'), sep=';')

    return xgb, dc, xg, bp


def build_meta_features(xgb_preds, dc, xg, bp):
    # Start from xg as base — it has match_id, season, home/away team, actual outcome
    base = xg[['match_id', 'season_name', 'matchday', 'home_team', 'away_team',
                'home_score', 'away_score', 'actual_outcome',
                'xgb_home_win', 'xgb_draw', 'xgb_away_win']].copy()
    base['match_id'] = base['match_id'].astype(int)

    # Merge DC probabilities from bundesliga_features — covers all seasons
    # filter to home rows only (one row per match)
    dc_from_features = dc[dc['team_id'] == dc['home_team_id']][
        ['match_id', 'dc_home_win', 'dc_draw', 'dc_away_win']
    ].copy()
    dc_from_features['match_id'] = dc_from_features['match_id'].astype(int)
    base = base.merge(dc_from_features, on='match_id', how='left')

    # Merge Bivariate Poisson probabilities
    bp_slim = bp[['match_id', 'bp_home_win', 'bp_draw', 'bp_away_win']].copy()
    bp_slim['match_id'] = bp_slim['match_id'].astype(int)
    base = base.merge(bp_slim, on='match_id', how='left')

    # Merge XGBoost classifier probabilities
    # xgb classifier has one row per team — filter to home rows only
    xgb_home = xgb_preds[xgb_preds['team_id'] == xgb_preds['home_team_id']][
        ['match_id', 'pred_home_win', 'pred_draw', 'pred_away_win']
    ].copy()
    xgb_home['match_id'] = xgb_home['match_id'].astype(int)
    base = base.merge(xgb_home, on='match_id', how='left')

    print(f'Meta features built — {len(base)} matches')
    print(base[get_meta_feature_cols()].isna().sum())
    return base


def get_meta_feature_cols():
    return [
        'xgb_home_win', 'xgb_draw', 'xgb_away_win',
        'dc_home_win', 'dc_draw', 'dc_away_win',
        'bp_home_win', 'bp_draw', 'bp_away_win',
        'pred_home_win', 'pred_draw', 'pred_away_win'
    ]


def rolling_train(meta_df, seasons):
    all_predictions = []
    le = LabelEncoder()
    le.fit(['away_win', 'draw', 'home_win'])

    feature_cols = get_meta_feature_cols()

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        predict_season = seasons[i]

        train = meta_df[meta_df['season_name'].isin(train_seasons)].fillna(1/3)
        predict = meta_df[meta_df['season_name'] == predict_season].fillna(1/3)

        train_x = train[feature_cols]
        train_y = le.transform(train['actual_outcome'])

        predict_x = predict[feature_cols]

        # Logistic regression meta-learner — learns optimal weighting of model outputs
        # C=1.0 is standard regularisation, max_iter=1000 for convergence
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(train_x, train_y)

        probs = meta.predict_proba(predict_x)

        predict_context = predict.copy()
        predict_context['stack_away_win'] = probs[:, 0]
        predict_context['stack_draw'] = probs[:, 1]
        predict_context['stack_home_win'] = probs[:, 2]
        predict_context['stack_predicted'] = le.inverse_transform(probs.argmax(axis=1))

        all_predictions.append(predict_context)
        print(f'Fold {i}: trained on {train_seasons}, predicted {predict_season}')

    return pd.concat(all_predictions, ignore_index=True), le


def predict_with_threshold(probs_df, h_col, d_col, a_col, draw_threshold):
    def predict(row):
        if row[d_col] >= draw_threshold:
            return 'draw'
        if row[h_col] >= row[a_col]:
            return 'home_win'
        return 'away_win'
    return probs_df.apply(predict, axis=1)


def optimise_threshold(predictions_df, h_col, d_col, a_col,
                       low=0.25, high=0.40, min_draw_recall=0.20, target_accuracy=0.47):
    best_threshold = low
    best_score = -1
    for threshold in np.linspace(low, high, 50):
        predicted = predict_with_threshold(predictions_df, h_col, d_col, a_col, threshold)
        report = classification_report(predictions_df['actual_outcome'], predicted,
                                       output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        draw_recall = report.get('draw', {}).get('recall', 0)
        if draw_recall < min_draw_recall or accuracy < target_accuracy:
            continue
        score = accuracy * 0.6 + draw_recall * 0.4
        if score > best_score:
            best_score = score
            best_threshold = threshold
    print(f'  Optimal threshold: {best_threshold:.4f} (score: {best_score:.4f})')
    return best_threshold


def eval_model(predictions_df):
    print('\n--- Stacking Model ---')
    labels = ['away_win', 'draw', 'home_win']
    best_threshold = optimise_threshold(predictions_df, 'stack_home_win', 'stack_draw', 'stack_away_win')
    predictions_df['stack_predicted'] = predict_with_threshold(
        predictions_df, 'stack_home_win', 'stack_draw', 'stack_away_win', best_threshold
    )
    print(classification_report(predictions_df['actual_outcome'],
                                predictions_df['stack_predicted'], labels=labels))

    cm = confusion_matrix(predictions_df['actual_outcome'],
                          predictions_df['stack_predicted'], labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    plt.title('Stacking Model — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'stacking_confusion_matrix.png'))
    plt.close()


def compare_models(predictions_df):
    print('\n--- Individual Model Comparison ---')
    labels = ['away_win', 'draw', 'home_win']
    feature_cols = get_meta_feature_cols()

    models = {
        'XGBoost Classifier': ('pred_home_win', 'pred_draw', 'pred_away_win'),
        'Dixon-Coles':        ('dc_home_win',   'dc_draw',   'dc_away_win'),
        'XGBoost xG':         ('xgb_home_win',  'xgb_draw',  'xgb_away_win'),
        'Bivariate Poisson':  ('bp_home_win',   'bp_draw',   'bp_away_win'),
        'Stacking':           ('stack_home_win', 'stack_draw', 'stack_away_win'),
    }

    rows = []
    for name, (h, d, a) in models.items():
        print(f'\n{name}:')
        threshold = optimise_threshold(predictions_df, h, d, a)
        preds = predict_with_threshold(predictions_df, h, d, a, threshold)
        report = classification_report(predictions_df['actual_outcome'], preds,
                                       output_dict=True, zero_division=0)
        rows.append({
            'model': name,
            'accuracy': round(report['accuracy'], 3),
            'draw_recall': round(report.get('draw', {}).get('recall', 0), 3),
            'home_recall': round(report.get('home_win', {}).get('recall', 0), 3),
            'away_recall': round(report.get('away_win', {}).get('recall', 0), 3),
            'macro_f1': round(report['macro avg']['f1-score'], 3),
        })

    comparison_df = pd.DataFrame(rows)
    print(comparison_df.to_string(index=False))
    return comparison_df


TEAM_NAME_MAP = {
    'Arminia Bielefeld': 'Bielefeld',
    'Bayer Leverkusen': 'Leverkusen',
    'Borussia Dortmund': 'Dortmund',
    'Borussia Mönchengladbach': "M'gladbach",
    'Cologne': 'FC Koln',
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'Greuther Fürth': 'Greuther Furth',
    'Hertha Berlin': 'Hertha',
    'Pauli': 'St Pauli',
    'Schalke': 'Schalke 04',
    'VfB Stuttgart': 'Stuttgart'
}


def save_combined(predictions_df):
    odds = pd.read_csv(os.path.join(DATA_DIR, 'bundesliga_odds.csv'))

    # Apply team name mapping to match odds format
    combined = predictions_df.copy()
    combined['home_team_odds'] = combined['home_team'].replace(TEAM_NAME_MAP)
    combined['away_team_odds'] = combined['away_team'].replace(TEAM_NAME_MAP)

    odds_slim = odds[['HomeTeam', 'AwayTeam', 'season_name',
                       'PSH', 'PSD', 'PSA',
                       'B365H', 'B365D', 'B365A',
                       'AvgH', 'AvgD', 'AvgA',
                       'MaxH', 'MaxD', 'MaxA']].copy()

    combined = combined.merge(
        odds_slim,
        left_on=['home_team_odds', 'away_team_odds', 'season_name'],
        right_on=['HomeTeam', 'AwayTeam', 'season_name'],
        how='left'
    )

    unmatched = combined['PSH'].isna().sum()
    print(f'Unmatched rows: {unmatched} / {len(combined)}')

    combined = combined.drop(columns=['home_team_odds', 'away_team_odds', 'HomeTeam', 'AwayTeam'])
    combined.to_csv(os.path.join(DATA_DIR, 'all_predictions.csv'), sep=';', index=False)
    print(f'Saved all_predictions.csv — {len(combined)} matches')
    return combined


def save_predictions(predictions_df):
    cols = ['match_id', 'season_name', 'matchday', 'home_team', 'away_team',
            'home_score', 'away_score', 'actual_outcome', 'stack_predicted',
            'stack_home_win', 'stack_draw', 'stack_away_win',
            'xgb_home_win', 'xgb_draw', 'xgb_away_win',
            'dc_home_win', 'dc_draw', 'dc_away_win',
            'bp_home_win', 'bp_draw', 'bp_away_win',
            'pred_home_win', 'pred_draw', 'pred_away_win']
    predictions_df[cols].to_csv(os.path.join(DATA_DIR, 'stacking_predictions.csv'), sep=';', index=False)
    print(f'Saved stacking_predictions.csv — {len(predictions_df)} matches')


if __name__ == '__main__':
    xgb, dc, xg, bp = load_predictions()

    # XGBoost classifier needs rolling predictions — run betting.py rolling_train
    from betting import rolling_train as xgb_rolling_train, df as features_df, BEST_PARAMS, seasons
    xgb_preds, le = xgb_rolling_train(features_df, seasons, BEST_PARAMS)

    meta_df = build_meta_features(xgb_preds, dc, xg, bp)
    predictions_df, le = rolling_train(meta_df, SEASONS)
    eval_model(predictions_df)
    comparison_df = compare_models(predictions_df)
    save_predictions(predictions_df)
    save_combined(predictions_df)

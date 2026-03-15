import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from scipy.special import comb
from scipy.stats import poisson
import math
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
VIS_DIR = os.path.join(SRC_DIR, '..', 'visualisations')

df = pd.read_csv(os.path.join(DATA_DIR, 'bundesliga_features.csv'))

FEATURE_COLS = [
    'prev_xgoals', 'prev_possession', 'prev_shots', 'prev_ontarget', 'prev_accuracy',
    'rolling_avg_goals', 'rolling_xg_performance', 'rolling_shot_conversion',
    'cumulative_points', 'win_streak', 'points_rank',
    'home_form_wins', 'home_form_goals', 'away_form_wins', 'away_form_goals',
    'h2h_points', 'elo_before', 'opponent_elo_before',
    'homeaway', 'matchday', 'dc_lambda'
]

SEASONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']

PARAM_GRID = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1]
}

PARAM_GRID_2 = {
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]
}


def transform_data(df, train_seasons, val_season, predict_season):
    df = df.copy()
    df['homeaway'] = (df['homeaway'] == 'Home').astype(int)

    # Build one row per match with home and away features side by side
    home_rows = df[df['homeaway'] == 1].copy()
    away_rows = df[df['homeaway'] == 0].copy()

    # Merge home and away on match_id so we have both perspectives per row
    merged = home_rows.merge(
        away_rows[['match_id'] + FEATURE_COLS],
        on='match_id',
        suffixes=('_home', '_away')
    )

    feature_cols_merged = [f'{c}_home' for c in FEATURE_COLS] + [f'{c}_away' for c in FEATURE_COLS]

    # Targets
    # λ1 = home goals, λ2 = away goals
    # λ3 proxy = min(home, away) — shared scoring component
    merged['target_home'] = merged['home_score']
    merged['target_away'] = merged['away_score']
    merged['target_lambda3'] = merged[['home_score', 'away_score']].min(axis=1)

    train = merged[merged['season_name'].isin(train_seasons)]
    val = merged[merged['season_name'] == val_season]
    predict = merged[merged['season_name'] == predict_season]

    train_x = train[feature_cols_merged].fillna(0)
    val_x = val[feature_cols_merged].fillna(0)
    predict_x = predict[feature_cols_merged].fillna(0)

    predict_context = predict[['match_id', 'season_name', 'matchday_home',
                                'home_score', 'away_score', 'dc_lambda_home', 'dc_lambda_away']].copy()
    predict_context = predict_context.rename(columns={'matchday_home': 'matchday'})

    # Get team names from original df
    home_names = df[df['homeaway'] == 1][['match_id', 'team_name']].rename(columns={'team_name': 'home_team'})
    away_names = df[df['homeaway'] == 0][['match_id', 'team_name']].rename(columns={'team_name': 'away_team'})
    predict_context = predict_context.merge(home_names, on='match_id', how='left')
    predict_context = predict_context.merge(away_names, on='match_id', how='left')

    return (train_x,
            train['target_home'], train['target_away'], train['target_lambda3'],
            val_x,
            val['target_home'], val['target_away'], val['target_lambda3'],
            predict_x, predict_context)


def scale(train_x, val_x, predict_x):
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    val_x_scaled = scaler.transform(val_x)
    predict_x_scaled = scaler.transform(predict_x)
    return train_x_scaled, val_x_scaled, predict_x_scaled


def optimal_params(train_x_scaled, train_y, param_grid, fixed_params={}):
    reg = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', **fixed_params)
    grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5,
                               scoring='neg_mean_absolute_error', verbose=0)
    grid_search.fit(train_x_scaled, train_y)
    print(f'Best params: {grid_search.best_params_}, MAE: {-grid_search.best_score_:.4f}')
    return {**fixed_params, **grid_search.best_params_}


def train_regressors(train_x_scaled, train_y_home, train_y_away, train_y_l3,
                     val_x_scaled, val_y_home, val_y_away, val_y_l3, params):
    regressors = {}
    for name, train_y, val_y in [
        ('lambda1', train_y_home, val_y_home),
        ('lambda2', train_y_away, val_y_away),
        ('lambda3', train_y_l3, val_y_l3)
    ]:
        reg = XGBRegressor(objective='reg:squarederror', early_stopping_rounds=10,
                           eval_metric='rmse', **params)
        reg.fit(train_x_scaled, train_y,
                eval_set=[(val_x_scaled, val_y)],
                verbose=False)
        regressors[name] = reg
        print(f'Trained {name} regressor')
    return regressors


def predict_lambdas(regressors, predict_x_scaled, predict_context):
    predict_context = predict_context.copy()
    # Clip at 0 — negative lambdas are invalid for Poisson
    predict_context['lambda1'] = np.clip(regressors['lambda1'].predict(predict_x_scaled), 0, None)
    predict_context['lambda2'] = np.clip(regressors['lambda2'].predict(predict_x_scaled), 0, None)
    predict_context['lambda3'] = np.clip(regressors['lambda3'].predict(predict_x_scaled), 0, None)
    # lambda3 must be less than both lambda1 and lambda2 for valid bivariate Poisson
    predict_context['lambda3'] = np.minimum(
        predict_context['lambda3'],
        np.minimum(predict_context['lambda1'], predict_context['lambda2']) * 0.9
    )
    return predict_context


def bivariate_poisson_prob(x, y, lambda1, lambda2, lambda3):
    # P(X=x, Y=y) for bivariate Poisson with parameters lambda1, lambda2, lambda3
    # X = Z1 + Z3, Y = Z2 + Z3 where Z1,Z2,Z3 are independent Poisson
    # Summation runs over k = 0 to min(x, y) shared goals
    total = 0
    for k in range(min(x, y) + 1):
        term = (comb(x, k, exact=True) *
                comb(y, k, exact=True) *
                math.factorial(k) *
                ((lambda3 / (lambda1 * lambda2)) ** k if lambda1 * lambda2 > 0 else 0) *
                poisson.pmf(x, lambda1) *
                poisson.pmf(y, lambda2))
        total += term
    return np.exp(-lambda3) * total


def build_score_matrix(lambda1, lambda2, lambda3, max_goals=10):
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i, j] = bivariate_poisson_prob(i, j, lambda1, lambda2, lambda3)
    # Renormalise to handle truncation at max_goals
    matrix /= matrix.sum()
    return matrix


def outcome_probs(score_matrix):
    home_win = np.tril(score_matrix, -1).sum()
    draw = np.trace(score_matrix)
    away_win = np.triu(score_matrix, 1).sum()
    return home_win, draw, away_win


def add_outcome_probs(matches_df):
    home_wins, draws, away_wins = [], [], []
    for _, row in matches_df.iterrows():
        matrix = build_score_matrix(row['lambda1'], row['lambda2'], row['lambda3'])
        h, d, a = outcome_probs(matrix)
        home_wins.append(h)
        draws.append(d)
        away_wins.append(a)

    matches_df = matches_df.copy()
    matches_df['bp_home_win'] = home_wins
    matches_df['bp_draw'] = draws
    matches_df['bp_away_win'] = away_wins
    return matches_df


def get_actual_outcome(row):
    if row['home_score'] > row['away_score']:
        return 'home_win'
    elif row['home_score'] == row['away_score']:
        return 'draw'
    else:
        return 'away_win'


def predict_with_threshold(matches_df, draw_threshold):
    def predict_outcome(row):
        if row['bp_draw'] >= draw_threshold:
            return 'draw'
        if row['bp_home_win'] >= row['bp_away_win']:
            return 'home_win'
        return 'away_win'
    return matches_df.apply(predict_outcome, axis=1)


def optimise_threshold(matches_df, low=0.25, high=0.40, min_draw_recall=0.20, target_accuracy=0.47):
    best_threshold = low
    best_score = -1
    candidates = np.linspace(low, high, 50)

    for threshold in candidates:
        predicted = predict_with_threshold(matches_df, threshold)
        report = classification_report(matches_df['actual_outcome'], predicted,
                                       output_dict=True, zero_division=0)
        accuracy = report['accuracy']
        draw_recall = report.get('draw', {}).get('recall', 0)

        if draw_recall < min_draw_recall or accuracy < target_accuracy:
            continue

        score = accuracy * 0.6 + draw_recall * 0.4
        if score > best_score:
            best_score = score
            best_threshold = threshold

    print(f'Optimal threshold: {best_threshold:.4f} (score: {best_score:.4f})')
    return best_threshold


def eval_model(matches_df):
    print('\n--- Outcome Prediction ---')
    matches_df['actual_outcome'] = matches_df.apply(get_actual_outcome, axis=1)
    best_threshold = optimise_threshold(matches_df)
    matches_df['predicted_outcome'] = predict_with_threshold(matches_df, best_threshold)

    labels = ['away_win', 'draw', 'home_win']
    print(classification_report(matches_df['actual_outcome'], matches_df['predicted_outcome'], labels=labels))

    cm = confusion_matrix(matches_df['actual_outcome'], matches_df['predicted_outcome'], labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    plt.title('Bivariate Poisson — Outcome Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'bp_confusion_matrix.png'))
    plt.close()

    return matches_df


def save_predictions(matches_df):
    cols = ['match_id', 'season_name', 'matchday', 'home_team', 'away_team',
            'home_score', 'away_score', 'actual_outcome', 'predicted_outcome',
            'lambda1', 'lambda2', 'lambda3',
            'bp_home_win', 'bp_draw', 'bp_away_win']
    matches_df[cols].to_csv(os.path.join(DATA_DIR, 'bp_predictions.csv'), sep=';', index=False)
    print(f'Saved bp_predictions.csv — {len(matches_df)} matches')


def rolling_train(df, seasons, params):
    all_matches = []

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        predict_season = seasons[i]

        if len(train_seasons) > 1:
            actual_train = train_seasons[:-1]
            val_season = train_seasons[-1]
        else:
            actual_train = train_seasons
            val_season = train_seasons[-1]

        (train_x, train_y_home, train_y_away, train_y_l3,
         val_x, val_y_home, val_y_away, val_y_l3,
         predict_x, predict_context) = transform_data(df, actual_train, val_season, predict_season)

        train_x_scaled, val_x_scaled, predict_x_scaled = scale(train_x, val_x, predict_x)

        regressors = train_regressors(
            train_x_scaled, train_y_home, train_y_away, train_y_l3,
            val_x_scaled, val_y_home, val_y_away, val_y_l3,
            params
        )

        predict_context = predict_lambdas(regressors, predict_x_scaled, predict_context)
        all_matches.append(predict_context)
        print(f'Fold {i}: trained on {actual_train}, val on {val_season}, predicted {predict_season}')

    matches_df = pd.concat(all_matches, ignore_index=True)
    matches_df = add_outcome_probs(matches_df)
    return matches_df


if __name__ == '__main__':
    # Grid search on first fold
    (train_x, train_y_home, train_y_away, train_y_l3,
     val_x, val_y_home, val_y_away, val_y_l3,
     predict_x, _) = transform_data(df, [SEASONS[0]], SEASONS[0], SEASONS[1])
    train_x_scaled, val_x_scaled, _ = scale(train_x, val_x, predict_x)

    print('--- Grid search for lambda1 (home goals) ---')
    best_params = optimal_params(train_x_scaled, train_y_home, PARAM_GRID)
    best_params = optimal_params(train_x_scaled, train_y_home, PARAM_GRID_2, fixed_params=best_params)

    matches_df = rolling_train(df, SEASONS, best_params)
    matches_df = eval_model(matches_df)
    save_predictions(matches_df)
    print(matches_df[['home_team', 'away_team', 'lambda1', 'lambda2', 'lambda3',
                       'bp_home_win', 'bp_draw', 'bp_away_win']].head(10))

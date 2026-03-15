import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
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
    # Encode homeaway as binary — Home=1, Away=0
    df = df.copy()
    df['homeaway'] = (df['homeaway'] == 'Home').astype(int)

    train = df[df['season_name'].isin(train_seasons)]
    val = df[df['season_name'] == val_season]
    predict = df[df['season_name'] == predict_season]

    train_x = train[FEATURE_COLS].fillna(0)
    train_y = train['scored']

    val_x = val[FEATURE_COLS].fillna(0)
    val_y = val['scored']

    predict_x = predict[FEATURE_COLS].fillna(0)

    # Save context for merging predictions back to matches
    predict_context = predict[['match_id', 'team_id', 'home_team_id', 'away_team_id',
                                'team_name', 'season_name', 'matchday',
                                'scored', 'home_score', 'away_score', 'dc_lambda', 'dc_mu']].copy()

    return train_x, train_y, val_x, val_y, predict_x, predict_context


def scale(train_x, val_x, predict_x):
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    val_x_scaled = scaler.transform(val_x)
    predict_x_scaled = scaler.transform(predict_x)
    return train_x_scaled, val_x_scaled, predict_x_scaled


def train_regressor(train_x_scaled, train_y, val_x_scaled, val_y, params):
    reg = XGBRegressor(
        objective='reg:squarederror',
        early_stopping_rounds=10,
        eval_metric='rmse',
        **params
    )
    reg.fit(
        train_x_scaled, train_y,
        eval_set=[(val_x_scaled, val_y)],
        verbose=False
    )
    return reg


def predict_goals(reg, predict_x_scaled, predict_context):
    # Clip predictions at 0 — negative goal counts make no sense
    lambdas = np.clip(reg.predict(predict_x_scaled), 0, None)
    predict_context = predict_context.copy()
    predict_context['xgb_lambda'] = lambdas
    return predict_context


def build_match_lambdas(predictions_df):
    # Split into home and away rows — each has its own predicted lambda
    home = predictions_df[predictions_df['team_id'] == predictions_df['home_team_id']].copy()
    away = predictions_df[predictions_df['team_id'] != predictions_df['home_team_id']].copy()

    home = home[['match_id', 'season_name', 'matchday', 'team_name',
                  'home_score', 'away_score', 'xgb_lambda', 'dc_lambda']].rename(columns={
        'team_name': 'home_team',
        'xgb_lambda': 'home_lambda',
        'dc_lambda': 'dc_home_lambda'
    })

    away = away[['match_id', 'xgb_lambda', 'dc_mu', 'team_name']].rename(columns={
        'team_name': 'away_team',
        'xgb_lambda': 'away_lambda',
        'dc_mu': 'dc_away_lambda'
    })

    matches = home.merge(away, on='match_id', how='inner')
    return matches


def dixon_coles_correction(home_score, away_score, home_lambda, away_lambda, rho):
    # Dixon-Coles correction factor τ for low-scoring outcomes
    # Inflates 0-0 and 1-1, deflates 1-0 and 0-1 to fix Poisson's draw underestimation
    if home_score == 0 and away_score == 0:
        return 1 - home_lambda * away_lambda * rho
    elif home_score == 1 and away_score == 0:
        return 1 + away_lambda * rho
    elif home_score == 0 and away_score == 1:
        return 1 + home_lambda * rho
    elif home_score == 1 and away_score == 1:
        return 1 - rho
    else:
        return 1.0


def poisson_outcome_probs(home_lambda, away_lambda, rho=-0.13, max_goals=10):
    # Build score matrix — P(home scores i) * P(away scores j)
    home_probs = poisson.pmf(np.arange(max_goals), home_lambda)
    away_probs = poisson.pmf(np.arange(max_goals), away_lambda)
    score_matrix = np.outer(home_probs, away_probs)

    # Apply Dixon-Coles correction to low-scoring cells
    for i in range(min(2, max_goals)):
        for j in range(min(2, max_goals)):
            score_matrix[i, j] *= dixon_coles_correction(i, j, home_lambda, away_lambda, rho)

    # Renormalise so probabilities still sum to 1
    score_matrix /= score_matrix.sum()

    home_win = np.tril(score_matrix, -1).sum()
    draw = np.trace(score_matrix)
    away_win = np.triu(score_matrix, 1).sum()

    return home_win, draw, away_win


def estimate_rho(matches_df):
    # Find rho that maximises likelihood of observed low-scoring outcomes in training data
    def neg_log_likelihood(rho):
        ll = 0
        for _, row in matches_df.iterrows():
            h, a = int(row['home_score']), int(row['away_score'])
            if h <= 1 and a <= 1:
                tau = dixon_coles_correction(h, a, row['home_lambda'], row['away_lambda'], rho)
                if tau > 0:
                    ll += np.log(tau)
        return -ll

    result = minimize_scalar(neg_log_likelihood, bounds=(-1.0, 0.0), method='bounded')
    print(f'Estimated rho: {result.x:.4f}')
    return result.x


def add_outcome_probs(matches_df, rho=-0.13):
    results = matches_df.apply(
        lambda r: poisson_outcome_probs(r['home_lambda'], r['away_lambda'], rho=rho), axis=1
    )
    matches_df = matches_df.copy()
    matches_df['xgb_home_win'] = [r[0] for r in results]
    matches_df['xgb_draw'] = [r[1] for r in results]
    matches_df['xgb_away_win'] = [r[2] for r in results]
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
        if row['xgb_draw'] >= draw_threshold:
            return 'draw'
        if row['xgb_home_win'] >= row['xgb_away_win']:
            return 'home_win'
        return 'away_win'
    return matches_df.apply(predict_outcome, axis=1)


def optimise_threshold(matches_df, low=0.25, high=0.40, min_draw_recall=0.20, target_accuracy=0.50):
    # Binary search for threshold that maximises combined score
    # while maintaining minimum draw recall and target accuracy
    best_threshold = low
    best_score = -1

    # Generate 50 candidate thresholds between bounds
    candidates = np.linspace(low, high, 50)

    for threshold in candidates:
        predicted = predict_with_threshold(matches_df, threshold)
        report = classification_report(matches_df['actual_outcome'], predicted, output_dict=True, zero_division=0)

        accuracy = report['accuracy']
        draw_recall = report.get('draw', {}).get('recall', 0)

        # Skip if below minimum constraints
        if draw_recall < min_draw_recall or accuracy < target_accuracy:
            continue

        # Combined score — accuracy weighted higher but draw recall must contribute
        score = accuracy * 0.6 + draw_recall * 0.4

        if score > best_score:
            best_score = score
            best_threshold = threshold

    print(f'Optimal threshold: {best_threshold:.4f} (score: {best_score:.4f})')
    return best_threshold


def eval_model(matches_df, predict_context):
    print('\n--- Goal Prediction (MAE / RMSE) ---')
    mae = mean_absolute_error(predict_context['scored'], predict_context['xgb_lambda'])
    rmse = np.sqrt(mean_squared_error(predict_context['scored'], predict_context['xgb_lambda']))
    print(f'MAE:  {mae:.3f}')
    print(f'RMSE: {rmse:.3f}')

    print('\n--- Outcome Prediction ---')
    matches_df['actual_outcome'] = matches_df.apply(get_actual_outcome, axis=1)

    best_threshold = optimise_threshold(matches_df, target_accuracy=0.47)
    matches_df['predicted_outcome'] = predict_with_threshold(matches_df, best_threshold)
    labels = ['away_win', 'draw', 'home_win']
    print(classification_report(matches_df['actual_outcome'], matches_df['predicted_outcome'], labels=labels))

    cm = confusion_matrix(matches_df['actual_outcome'], matches_df['predicted_outcome'], labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    plt.title('XGBoost xG Model — Outcome Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'xg_confusion_matrix.png'))
    plt.close()

    return matches_df


def save_predictions(matches_df):
    cols = ['match_id', 'season_name', 'matchday', 'home_team', 'away_team',
            'home_score', 'away_score', 'actual_outcome', 'predicted_outcome',
            'home_lambda', 'away_lambda',
            'xgb_home_win', 'xgb_draw', 'xgb_away_win',
            'dc_home_lambda', 'dc_away_lambda']
    matches_df[cols].to_csv(os.path.join(DATA_DIR, 'xg_predictions.csv'), sep=';', index=False)
    print(f'Saved xg_predictions.csv — {len(matches_df)} matches')


def plot_lambda_distribution(matches_df):
    plt.figure(figsize=(10, 5))
    plt.hist(matches_df['home_lambda'], bins=30, alpha=0.5, label='XGB home λ')
    plt.hist(matches_df['dc_home_lambda'], bins=30, alpha=0.5, label='DC home λ')
    plt.xlabel('Predicted goals (λ)')
    plt.ylabel('Frequency')
    plt.title('XGBoost vs Dixon-Coles λ distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'lambda_distribution.png'))
    plt.close()
    print('Saved lambda_distribution.png')


def optimal_params(train_x_scaled, train_y, param_grid, fixed_params={}):
    reg = XGBRegressor(objective='reg:squarederror', eval_metric='rmse', **fixed_params)
    grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=2)
    grid_search.fit(train_x_scaled, train_y)
    print(f'Best params: {grid_search.best_params_}')
    print(f'Best MAE: {-grid_search.best_score_:.4f}')
    return {**fixed_params, **grid_search.best_params_}


def rolling_train(df, seasons, params):
    all_predictions = []
    all_contexts = []

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        predict_season = seasons[i]

        if len(train_seasons) > 1:
            actual_train = train_seasons[:-1]
            val_season = train_seasons[-1]
        else:
            actual_train = train_seasons
            val_season = train_seasons[-1]  # use same season for val on first fold

        train_x, train_y, val_x, val_y, predict_x, predict_context = transform_data(
            df, actual_train, val_season, predict_season
        )
        train_x_scaled, val_x_scaled, predict_x_scaled = scale(train_x, val_x, predict_x)
        reg = train_regressor(train_x_scaled, train_y, val_x_scaled, val_y, params)
        predict_context = predict_goals(reg, predict_x_scaled, predict_context)

        all_predictions.append(predict_context)
        print(f'Fold {i}: trained on {actual_train}, val on {val_season}, predicted {predict_season}')

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    matches_df = build_match_lambdas(predictions_df)
    return matches_df, predictions_df


if __name__ == '__main__':
    # Run grid search on first training fold to find optimal params
    train_x, train_y, val_x, val_y, _, _ = transform_data(df, [SEASONS[0]], SEASONS[0], SEASONS[1])
    train_x_scaled, val_x_scaled, _ = scale(train_x, val_x, val_x)
    best_params = optimal_params(train_x_scaled, train_y, PARAM_GRID)
    best_params = optimal_params(train_x_scaled, train_y, PARAM_GRID_2, fixed_params=best_params)

    matches_df, predictions_df = rolling_train(df, SEASONS, best_params)
    rho = estimate_rho(matches_df)
    matches_df = add_outcome_probs(matches_df, rho=rho)
    matches_df = eval_model(matches_df, predictions_df)
    save_predictions(matches_df)
    plot_lambda_distribution(matches_df)
    print(matches_df[['home_team', 'away_team', 'home_lambda', 'away_lambda',
                       'xgb_home_win', 'xgb_draw', 'xgb_away_win']].head(10))

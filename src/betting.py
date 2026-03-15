import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.isotonic import IsotonicRegression

# Resolve paths relative to this file so imports work from any directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
VIS_DIR = os.path.join(SRC_DIR, '..', 'visualisations')

# Data Frames
df = pd.read_csv(os.path.join(DATA_DIR, 'bundesliga_features.csv'), sep=',')
df_dc = pd.read_csv(os.path.join(DATA_DIR, 'dc_predictions.csv'), sep=';')
df_odds = pd.read_csv(os.path.join(DATA_DIR, 'bundesliga_odds.csv'), sep=',')

# Variables
BEST_PARAMS = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.01}
seasons = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']

# Maps our scraped team names to football-data.co.uk names
# Teams not listed here already match exactly (e.g. Augsburg, Mainz)
TEAM_NAME_MAP = {
    'Arminia Bielefeld': 'Bielefeld',
    'Bayer Leverkusen': 'Leverkusen',
    'Borussia Dortmund': 'Dortmund',
    'Borussia M\xf6nchengladbach': "M'gladbach",
    'Cologne': 'FC Koln',
    'Eintracht Frankfurt': 'Ein Frankfurt',
    'Greuther F\xfcrth': 'Greuther Furth',
    'Hertha Berlin': 'Hertha',
    'Pauli': 'St Pauli',
    'Schalke': 'Schalke 04',
    'VfB Stuttgart': 'Stuttgart'
}


# Functions
def get_outcome(row):
    if row['scored'] > row['conceded']:
        return 'home_win'
    elif row['scored'] == row['conceded']:
        return 'draw'
    else:
        return 'away_win'

def remove_leak_identifier(df):
    return df.drop(["match_id", "team_id", "home_team_id", "away_team_id", "opponent_id", "season_name", "team_name", "matchday", "homeaway",
                    "scored", "conceded", "xgoals", "passes", "accuracy", "shots", "ontarget", "offtarget", "possession", "tackleswon",
                    "corners", "offsides", "fouls", "home_score", "away_score", "goal_difference", "win", "dc_lambda", "dc_mu", "dc_home_win", "dc_draw", "dc_away_win"
], axis='columns')

def rolling_train(df, seasons, params):
    all_predictions = []
    df['outcome'] = df.apply(get_outcome, axis=1)

    # Fit le once on full dataset so classes are consistent across all folds
    le = LabelEncoder()
    le.fit(df['outcome'])

    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        predict_season = seasons[i]

        # Use last training season as validation for early stopping if more than one season available
        if len(train_seasons) > 1:
            actual_train_seasons = train_seasons[:-1]
            val_season = train_seasons[-1]
        else:
            actual_train_seasons = train_seasons
            val_season = None

        train_x = df[df['season_name'].isin(actual_train_seasons)].pipe(remove_leak_identifier).drop('outcome', axis='columns')
        train_y = df[df['season_name'].isin(actual_train_seasons)]['outcome']

        # Save context before dropping identifiers
        predict_df = df[df['season_name'] == predict_season]
        predict_context = predict_df[['match_id', 'team_id', 'home_team_id', 'away_team_id', 'team_name', 'season_name', 'matchday', 'home_score', 'away_score', 'outcome']].copy()
        predict_x = predict_df.pipe(remove_leak_identifier).drop('outcome', axis='columns')

        scaler = StandardScaler()
        train_x_scaled = scaler.fit_transform(train_x)
        predict_x_scaled = scaler.transform(predict_x)

        train_y_encoded = le.transform(train_y)

        sample_weights = compute_sample_weight(class_weight='balanced', y=train_y_encoded)

        if val_season:
            clf_xgb = XGBClassifier(
                objective='multi:softprob',
                early_stopping_rounds=10,
                eval_metric='mlogloss',
                **params
            )
            val_x = df[df['season_name'] == val_season].pipe(remove_leak_identifier).drop('outcome', axis='columns')
            val_y = df[df['season_name'] == val_season]['outcome']
            val_x_scaled = scaler.transform(val_x)
            val_y_encoded = le.transform(val_y)
            clf_xgb.fit(
                train_x_scaled, train_y_encoded,
                sample_weight=sample_weights,
                eval_set=[(val_x_scaled, val_y_encoded)],
                verbose=False
            )
            # Calibrate on val set — fit one isotonic regressor per class
            val_probs = clf_xgb.predict_proba(val_x_scaled)
            n_classes = val_probs.shape[1]
            calibrators = []
            for i in range(n_classes):
                ir = IsotonicRegression(out_of_bounds='clip')
                ir.fit(val_probs[:, i], (val_y_encoded == i).astype(int))
                calibrators.append(ir)
        else:
            clf_xgb = XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                **params
            )
            clf_xgb.fit(train_x_scaled, train_y_encoded, sample_weight=sample_weights)
            calibrators = None  # no calibration on first fold

        raw_probs = clf_xgb.predict_proba(predict_x_scaled)
        if calibrators:
            probs = np.column_stack([c.predict(raw_probs[:, i]) for i, c in enumerate(calibrators)])
            probs = probs / probs.sum(axis=1, keepdims=True)
        else:
            probs = raw_probs

        # Attach probabilities — le.classes_ gives alphabetical order [away_win, draw, home_win]
        predict_context = predict_context.copy()
        predict_context['pred_away_win'] = probs[:, 0]
        predict_context['pred_draw'] = probs[:, 1]
        predict_context['pred_home_win'] = probs[:, 2]

        all_predictions.append(predict_context)
        print(f'Fold {i}: trained on {actual_train_seasons}, val on {val_season}, predicted {predict_season}')

    return pd.concat(all_predictions, ignore_index=True), le

def eval_rolling(predictions_df, le):
    print('\n--- Rolling Window Evaluation per Season ---')
    all_actual = []
    all_predicted = []

    for season, group in predictions_df.groupby('season_name'):
        actual = group['outcome']
        predicted = group[['pred_away_win', 'pred_draw', 'pred_home_win']].apply(
            lambda r: le.classes_[r.argmax()], axis=1
        )
        all_actual.extend(actual)
        all_predicted.extend(predicted)
        print(f'\n{season}:')
        print(classification_report(actual, predicted, target_names=le.classes_))

    # Overall confusion matrix across all seasons
    cm = confusion_matrix(all_actual, all_predicted, labels=le.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot()
    plt.title('Rolling Window — Overall Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'rolling_confusion_matrix.png'))
    plt.close()

def add_dc_probabilities(predictions_df, df_dc):
    # Filter to home perspective only — one row per match, aligns with odds data
    home_preds = predictions_df[predictions_df['team_id'] == predictions_df['home_team_id']].copy()

    # Merge DC probabilities on match_id
    home_preds = home_preds.merge(
        df_dc[['match_id', 'home_win', 'draw', 'away_win']],
        on='match_id',
        how='left'
    )
    home_preds = home_preds.rename(columns={
        'home_win': 'dc_home_win',
        'draw': 'dc_draw',
        'away_win': 'dc_away_win'
    })
    return home_preds


def merge_odds(home_preds, df_odds, df):
    # Build a team_id -> team_name lookup from the full df (has both home and away rows)
    team_lookup = df[['team_id', 'team_name']].drop_duplicates().set_index('team_id')['team_name']

    # Resolve names for both home and away teams, then apply TEAM_NAME_MAP where names differ
    home_preds = home_preds.copy()
    home_preds['home_team_name'] = home_preds['home_team_id'].map(team_lookup).replace(TEAM_NAME_MAP)
    home_preds['away_team_name'] = home_preds['away_team_id'].map(team_lookup).replace(TEAM_NAME_MAP)

    # Slim the odds df down to only the bookmakers we care about
    odds_cols = ['HomeTeam', 'AwayTeam', 'season_name',
                 'B365H', 'B365D', 'B365A',
                 'PSH', 'PSD', 'PSA',
                 'AvgH', 'AvgD', 'AvgA',
                 'MaxH', 'MaxD', 'MaxA']
    df_odds_slim = df_odds[odds_cols].copy()

    # Left join on home name, away name, and season — unmatched rows get NaN odds
    merged = home_preds.merge(
        df_odds_slim,
        left_on=['home_team_name', 'away_team_name', 'season_name'],
        right_on=['HomeTeam', 'AwayTeam', 'season_name'],
        how='left'
    )

    # Print unmatched count so we can extend TEAM_NAME_MAP if needed
    unmatched = merged['B365H'].isna().sum()
    print(f'Unmatched rows (no odds found): {unmatched} / {len(merged)}')

    return merged

def calc_implied_prob(df):
    total = (1/df['PSH']) + (1/df['PSD']) + (1/df['PSA'])
    df['fair_h'] = (1/df['PSH']) / total
    df['fair_d'] = (1/df['PSD']) / total
    df['fair_a'] = (1/df['PSA']) / total
    return df

def calc_edge(df):
    df['edge_h'] = df['pred_home_win'] - df['fair_h']
    df['edge_d'] = df['pred_draw']    - df['fair_d']
    df['edge_a'] = df['pred_away_win'] - df['fair_a']
    return df

def calc_ev(df):
    df['ev_h'] = df['pred_home_win'] * (df['PSH'] - 1) - (1 - df['pred_home_win'])
    df['ev_d'] = df['pred_draw']     * (df['PSD'] - 1) - (1 - df['pred_draw'])
    df['ev_a'] = df['pred_away_win'] * (df['PSA'] - 1) - (1 - df['pred_away_win'])
    return df

def flag_bets(df):
    # Exclude heavy underdogs — model unreliable on extreme mismatches (implied prob < 10%)
    valid_home = df['ev_h'] > 0.20
    valid_away = (df['ev_a'] > 0.20) & (df['PSA'] <= 10)

    df['bet_flag'] = (valid_home | valid_away).astype(int)
    df['bet_outcome'] = None
    df.loc[valid_home, 'bet_outcome'] = 'home_win'
    df.loc[valid_away, 'bet_outcome'] = 'away_win'

    return df


def calc_kelly(df, fraction=0.25, max_bet=0.10):
    b_h = df['PSH'] - 1
    df['kelly_h'] = fraction * ((b_h * df['pred_home_win'] - (1 - df['pred_home_win'])) / b_h)

    b_a = df['PSA'] - 1
    df['kelly_a'] = fraction * ((b_a * df['pred_away_win'] - (1 - df['pred_away_win'])) / b_a)

    # Negative Kelly means no edge — floor at 0, cap at max_bet
    df['kelly_h'] = df['kelly_h'].clip(lower=0, upper=max_bet)
    df['kelly_a'] = df['kelly_a'].clip(lower=0, upper=max_bet)

    return df


def simulate_bankroll(df, starting_bankroll=10000):
    # Filter to flagged bets only and sort chronologically
    bets = df[df['bet_flag'] == 1].copy()
    bets = bets.sort_values(['season_name', 'matchday']).reset_index(drop=True)

    bankroll = starting_bankroll
    history = [{'season_name': None, 'matchday': None, 'bankroll': bankroll, 'bet': 'start'}]

    for _, row in bets.iterrows():
        outcome = row['bet_outcome']

        if outcome == 'home_win':
            stake = bankroll * row['kelly_h']
            won = row['outcome'] == 'home_win'
            odds = row['PSH']
        else:
            stake = bankroll * row['kelly_a']
            won = row['outcome'] == 'away_win'
            odds = row['PSA']

        if won:
            bankroll += stake * (odds - 1)
        else:
            bankroll -= stake

        history.append({
            'season_name': row['season_name'],
            'matchday': row['matchday'],
            'bankroll': bankroll,
            'bet': outcome
        })

    history_df = pd.DataFrame(history)

    # Line chart
    plt.figure(figsize=(12, 5))
    plt.plot(history_df.index, history_df['bankroll'])
    plt.axhline(starting_bankroll, color='grey', linestyle='--', label='Starting bankroll')
    plt.title('Bankroll Development')
    plt.xlabel('Bet number')
    plt.ylabel('Bankroll (€)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'bankroll.png'))
    plt.close()

    print(f'Final bankroll: €{bankroll:.2f}')
    print(f'Return: {((bankroll - starting_bankroll) / starting_bankroll) * 100:.1f}%')
    return history_df


def calc_edge_dc(df):
    df['dc_edge_h'] = df['dc_home_win'] - df['fair_h']
    df['dc_edge_d'] = df['dc_draw']     - df['fair_d']
    df['dc_edge_a'] = df['dc_away_win'] - df['fair_a']
    return df

def calc_ev_dc(df):
    df['dc_ev_h'] = df['dc_home_win'] * (df['PSH'] - 1) - (1 - df['dc_home_win'])
    df['dc_ev_a'] = df['dc_away_win'] * (df['PSA'] - 1) - (1 - df['dc_away_win'])
    return df

def flag_bets_dc(df):
    valid_home = df['dc_ev_h'] > 0.20
    valid_away = (df['dc_ev_a'] > 0.20) & (df['PSA'] <= 10)
    df['dc_bet_flag'] = (valid_home | valid_away).astype(int)
    df['dc_bet_outcome'] = None
    df.loc[valid_home, 'dc_bet_outcome'] = 'home_win'
    df.loc[valid_away, 'dc_bet_outcome'] = 'away_win'
    return df

def calc_kelly_dc(df, fraction=0.25, max_bet=0.10):
    b_h = df['PSH'] - 1
    df['dc_kelly_h'] = (fraction * ((b_h * df['dc_home_win'] - (1 - df['dc_home_win'])) / b_h)).clip(0, max_bet)
    b_a = df['PSA'] - 1
    df['dc_kelly_a'] = (fraction * ((b_a * df['dc_away_win'] - (1 - df['dc_away_win'])) / b_a)).clip(0, max_bet)
    return df

def simulate_bankroll_dc(df, starting_bankroll=10000):
    bets = df[df['dc_bet_flag'] == 1].copy()
    bets = bets.sort_values(['season_name', 'matchday']).reset_index(drop=True)

    bankroll = starting_bankroll
    history = [{'season_name': None, 'matchday': None, 'bankroll': bankroll, 'bet': 'start'}]

    for _, row in bets.iterrows():
        outcome = row['dc_bet_outcome']

        if outcome == 'home_win':
            stake = bankroll * row['dc_kelly_h']
            won = row['outcome'] == 'home_win'
            odds = row['PSH']
        else:
            stake = bankroll * row['dc_kelly_a']
            won = row['outcome'] == 'away_win'
            odds = row['PSA']

        if won:
            bankroll += stake * (odds - 1)
        else:
            bankroll -= stake

        history.append({
            'season_name': row['season_name'],
            'matchday': row['matchday'],
            'bankroll': bankroll,
            'bet': outcome
        })

    history_df = pd.DataFrame(history)

    plt.figure(figsize=(12, 5))
    plt.plot(history_df.index, history_df['bankroll'])
    plt.axhline(starting_bankroll, color='grey', linestyle='--', label='Starting bankroll')
    plt.title('Bankroll Development — Dixon-Coles')
    plt.xlabel('Bet number')
    plt.ylabel('Bankroll (€)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'bankroll_dc.png'))
    plt.close()

    print(f'DC Final bankroll: €{bankroll:.2f}')
    print(f'DC Return: {((bankroll - starting_bankroll) / starting_bankroll) * 100:.1f}%')
    return history_df


if __name__ == '__main__':
    predictions_df, le = rolling_train(df, seasons, BEST_PARAMS)
    eval_rolling(predictions_df, le)
    predictions_df = add_dc_probabilities(predictions_df, df_dc)
    predictions_df = merge_odds(predictions_df, df_odds, df)
    predictions_df = predictions_df.pipe(calc_implied_prob).pipe(calc_edge).pipe(calc_ev).pipe(flag_bets).pipe(calc_kelly)
    print(f'Total EV positive bets: {len(predictions_df[predictions_df["bet_flag"] == 1])}')
    bankroll_df = simulate_bankroll(predictions_df)

    predictions_df = predictions_df.pipe(calc_edge_dc).pipe(calc_ev_dc).pipe(flag_bets_dc).pipe(calc_kelly_dc)
    print(f'Total DC EV positive bets: {len(predictions_df[predictions_df["dc_bet_flag"] == 1])}')
    bankroll_dc_df = simulate_bankroll_dc(predictions_df)

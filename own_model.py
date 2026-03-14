# 1. EDA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 2. XGBoost Model
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('bundesliga_features.csv')

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
                    "corners", "offsides", "fouls", "home_score", "away_score", "goal_difference", "win",
], axis='columns')

def transform_data(df):
    df["outcome"] = df.apply(get_outcome, axis=1)
    train_x = df[~df['season_name'].isin(['2024-2025', '2025-2026'])].pipe(remove_leak_identifier).drop('outcome', axis='columns')
    train_y = df[~df['season_name'].isin(['2024-2025', '2025-2026'])]['outcome']
    validate_x = df[df['season_name'] == '2024-2025'].pipe(remove_leak_identifier).drop('outcome', axis='columns')
    validate_y = df[df['season_name'] == '2024-2025']['outcome']
    test_x = df[df['season_name'] == '2025-2026'].pipe(remove_leak_identifier).drop('outcome', axis='columns')
    test_y = df[df['season_name'] == '2025-2026']['outcome']
    return train_x, train_y, validate_x, validate_y, test_x, test_y

def scaled_corr_matrix(train_x, validate_x, test_x):
    # Corr shows 0.83 between goal difference and elo before, drop goal difference
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    validate_x_scaled = scaler.transform(validate_x)
    test_x_scaled = scaler.transform(test_x)
    train_scaled_df = pd.DataFrame(train_x_scaled, columns=train_x.columns)
    corr = train_scaled_df.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    plt.show()
    return train_x_scaled, validate_x_scaled, test_x_scaled

def encode_dependents(train_y, validate_y, test_y):
    le = LabelEncoder()
    train_y_encoded = le.fit_transform(train_y)
    validate_y_encoded = le.transform(validate_y)
    test_y_encoded = le.transform(test_y)
    return train_y_encoded, validate_y_encoded, test_y_encoded, le

def train_model(train_x_scaled, train_y_encoded, validate_x_scaled, validate_y_encoded, test_x_scaled):
    clf_xgb = XGBClassifier(
        objective='multi:softprob',
        early_stopping_rounds=10,
        eval_metric='mlogloss'
    )
    clf_xgb.fit(
        train_x_scaled, train_y_encoded,
        verbose=True,
        eval_set=[(validate_x_scaled, validate_y_encoded)]
    )
    preds = clf_xgb.predict(test_x_scaled)
    return clf_xgb, preds

def eval_model(le, preds, test_y_encoded):
    print(classification_report(test_y_encoded, preds, target_names=le.classes_))
    cm = confusion_matrix(test_y_encoded, preds, labels=list(range(len(le.classes_))))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot()
    plt.tight_layout()
    plt.savefig('eval_matrix.png')
    plt.show()


if __name__ == '__main__':
    train_x, train_y, validate_x, validate_y, test_x, test_y = transform_data(df)
    train_x_scaled, validate_x_scaled, test_x_scaled = scaled_corr_matrix(train_x, validate_x, test_x)
    train_y_enc, validate_y_enc, test_y_enc, le = encode_dependents(train_y, validate_y, test_y)
    clf_xgb, preds = train_model(train_x_scaled, train_y_enc, validate_x_scaled, validate_y_enc, test_x_scaled)
    eval_model(le, preds, test_y_enc)

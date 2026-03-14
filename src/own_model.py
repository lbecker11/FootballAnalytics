import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import shap

# Var declaration
df = pd.read_csv('../data/bundesliga_features.csv')
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1]}
param_grid_2 = {
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5]}
# Function declarations
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
    return train_x_scaled, validate_x_scaled, test_x_scaled

def encode_dependents(train_y, validate_y, test_y):
    le = LabelEncoder()
    train_y_encoded = le.fit_transform(train_y)
    validate_y_encoded = le.transform(validate_y)
    test_y_encoded = le.transform(test_y)
    return train_y_encoded, validate_y_encoded, test_y_encoded, le

def train_model(train_x_scaled, train_y_encoded, validate_x_scaled, validate_y_encoded, test_x_scaled, best_params):
    clf_xgb = XGBClassifier(
        objective='multi:softprob',
        early_stopping_rounds=10,
        eval_metric='mlogloss',
        **best_params
    )
    sample_weights = compute_sample_weight(class_weight='balanced', y=train_y_encoded)
    clf_xgb.fit(
        train_x_scaled, train_y_encoded,
        sample_weight=sample_weights,
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
    plt.savefig('../visualisations/eval_matrix.png')
    plt.show()

def prep_for_cv(train_x_scaled, validate_x_scaled, train_y_enc, validate_y_enc):
    X_search = np.concatenate([train_x_scaled, validate_x_scaled])
    y_search = np.concatenate([train_y_enc, validate_y_enc])
    return X_search, y_search

def optimal_params(X_search, y_search, params_, fixed_params={}):
    clf = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', **fixed_params)
    grid_search = GridSearchCV(estimator=clf, param_grid=params_, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X_search, y_search)
    print(f'Best params: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_:.4f}')
    return grid_search.best_params_

def get_dc_features(df, seasons_exclude):
    # Extract DC probabilities as features — kept separate from XGBoost features
    # so the meta-learner can learn to combine both models
    dc_cols = ['dc_home_win', 'dc_draw', 'dc_away_win']
    train_dc = df[~df['season_name'].isin(seasons_exclude)][dc_cols].values
    validate_dc = df[df['season_name'] == '2024-2025'][dc_cols].values
    test_dc = df[df['season_name'] == '2025-2026'][dc_cols].values
    return train_dc, validate_dc, test_dc

def build_stacking_input(x_scaled, dc_features):
    # Concatenate XGBoost features with DC probabilities
    # The meta-learner sees both sets of inputs and learns the optimal combination
    return np.concatenate([x_scaled, dc_features], axis=1)

def train_stacking_model(train_x_stack, train_y_enc, validate_x_stack, validate_y_enc, test_x_stack, best_params):
    # Base estimator: XGBoost trained on engineered features + DC features
    xgb = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        **best_params
    )
    # Meta-learner: logistic regression learns how to combine XGBoost's
    # cross-validated out-of-fold probabilities into a final prediction
    # cv=5 means XGBoost generates predictions on held-out folds so the
    # meta-learner never trains on predictions the base model already saw
    stack = StackingClassifier(
        estimators=[('xgb', xgb)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        stack_method='predict_proba'
    )
    sample_weights = compute_sample_weight(class_weight='balanced', y=train_y_enc)
    stack.fit(train_x_stack, train_y_enc, sample_weight=sample_weights)
    stack_preds = stack.predict(test_x_stack)
    return stack, stack_preds

def eval_stacking_model(le, stack_preds, test_y_enc):
    print('\n--- Stacking Model ---')
    print(classification_report(test_y_enc, stack_preds, target_names=le.classes_))
    cm = confusion_matrix(test_y_enc, stack_preds, labels=list(range(len(le.classes_))))
    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot()
    plt.tight_layout()
    plt.savefig('../visualisations/stacking_matrix.png')
    plt.show()

def plot_roc_comparison(le, test_y_enc, model_probs_dict):
    # Binarize the true labels — converts [0,1,2] into a 3-column matrix
    # e.g. class 1 becomes [0,1,0] so we can compute one ROC curve per class
    classes = list(range(len(le.classes_)))
    y_bin = label_binarize(test_y_enc, classes=classes)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loop over each outcome class
    for i, class_name in enumerate(le.classes_):
        ax = axes[i]
        # Loop over each model and plot its ROC curve for this class
        for model_name, probs in model_probs_dict.items():
            # roc_curve computes false positive rate and true positive rate
            # at every possible probability threshold
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            # auc computes the area under the curve — higher is better
            # 0.5 = random guessing, 1.0 = perfect
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.2f})')

        # Diagonal line represents random guessing (AUC=0.5)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_title(f'ROC — {class_name}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()

    plt.tight_layout()
    plt.savefig('../visualisations/roc_comparison.png')
    plt.show()

def save_model_plus_params(clf_xgb, test_x, test_x_scaled, preds, test_y_enc, le):
    clf_xgb.save_model('../models/xgb_model.json')
    probs = clf_xgb.predict_proba(test_x_scaled)
    test_preds_df = test_x.copy()
    test_preds_df['pred_away_win'] = probs[:, 0]
    test_preds_df['pred_draw'] = probs[:, 1]
    test_preds_df['pred_home_win'] = probs[:, 2]
    test_preds_df['predicted_outcome'] = le.inverse_transform(preds)
    test_preds_df['actual_outcome'] = le.inverse_transform(test_y_enc)
    test_preds_df.to_csv('../data/xgb_predictions.csv', sep=';', index=False)

def compute_shap(clf_xgb, test_x_scaled, test_x, le):
    explainer = shap.TreeExplainer(clf_xgb)
    shap_values = explainer(test_x_scaled)
    np.save('../models/shap_values.npy', shap_values.values)

    # Loop over each outcome class and produce beeswarm and bar plots
    for i, class_name in enumerate(le.classes_):
        # Slice SHAP values for this class — shape (n_samples, n_features)
        # Add feature names so plots show column names instead of indices
        class_shap = shap.Explanation(
            values=shap_values.values[:, :, i],
            base_values=shap_values.base_values[:, i],
            data=test_x.values,
            feature_names=test_x.columns.tolist()
        )

        # Beeswarm — each dot is one match, x-axis is SHAP value impact,
        # colour shows whether the feature value was high (red) or low (blue)
        plt.figure()
        shap.plots.beeswarm(class_shap, show=False)
        plt.title(f'SHAP Beeswarm — {class_name}')
        plt.tight_layout()
        plt.savefig(f'../visualisations/shap_beeswarm_{class_name}.png')
        plt.show()

        # Bar plot — mean absolute SHAP value per feature,
        # cleanest summary of global feature importance
        plt.figure()
        shap.plots.bar(class_shap, show=False)
        plt.title(f'SHAP Feature Importance — {class_name}')
        plt.tight_layout()
        plt.savefig(f'../visualisations/shap_bar_{class_name}.png')
        plt.show()
    
if __name__ == '__main__':
    train_x, train_y, validate_x, validate_y, test_x, test_y = transform_data(df)
    train_x_scaled, validate_x_scaled, test_x_scaled = scaled_corr_matrix(train_x, validate_x, test_x)
    train_y_enc, validate_y_enc, test_y_enc, le = encode_dependents(train_y, validate_y, test_y)
    X_search, y_search = prep_for_cv(train_x_scaled, validate_x_scaled, train_y_enc, validate_y_enc)
    best_params = optimal_params(X_search, y_search, param_grid)
    clf_xgb, preds = train_model(train_x_scaled, train_y_enc, validate_x_scaled, validate_y_enc, test_x_scaled, best_params)
    eval_model(le, preds, test_y_enc)
    save_model_plus_params(clf_xgb, test_x, test_x_scaled, preds, test_y_enc, le)

    # Stacking model — combines XGBoost features with Dixon-Coles probabilities
    train_dc, validate_dc, test_dc = get_dc_features(df, ['2024-2025', '2025-2026'])
    train_x_stack = build_stacking_input(train_x_scaled, train_dc)
    validate_x_stack = build_stacking_input(validate_x_scaled, validate_dc)
    test_x_stack = build_stacking_input(test_x_scaled, test_dc)
    stack, stack_preds = train_stacking_model(train_x_stack, train_y_enc, validate_x_stack, validate_y_enc, test_x_stack, best_params)
    eval_stacking_model(le, stack_preds, test_y_enc)

    # ROC AUC comparison — filter to home rows only to match DC predictions (one row per match)
    test_full = df[df['season_name'] == '2025-2026']
    home_mask = (test_full['team_id'] == test_full['home_team_id']).values

    xgb_probs = clf_xgb.predict_proba(test_x_scaled)
    stack_probs = stack.predict_proba(test_x_stack)

    # Load Dixon-Coles probabilities from saved predictions
    dc_preds = pd.read_csv('../data/dc_predictions.csv', sep=';')
    dc_probs = dc_preds[['away_win', 'draw', 'home_win']].values

    model_probs_dict = {
        'XGBoost': xgb_probs[home_mask],
        'Stacking': stack_probs[home_mask],
        'Dixon-Coles': dc_probs
    }
    plot_roc_comparison(le, test_y_enc[home_mask], model_probs_dict)

    # SHAP analysis
    compute_shap(clf_xgb, test_x_scaled, test_x, le)
    


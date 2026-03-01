import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('bundesliga_features.csv', sep=',')
train = df[df['season_name'] != '2025-2026']
test = df[df['season_name'] == '2025-2026']

# Get unique teams and map to index
teams = sorted(df['team_name'].unique())
team_idx = {team: i for i, team in enumerate(teams)}
n_teams = len(teams)

# Build one row per match for training (home perspective only)
home_train = train[train['team_id'] == train['home_team_id']].copy()
home_train['home_team_name'] = home_train['team_name']
home_train = home_train.merge(
    train[train['team_id'] == train['away_team_id']][['match_id', 'team_name']],
    on='match_id', suffixes=('', '_away')
)
home_train = home_train.rename(columns={'team_name_away': 'away_team_name'})

def tau(x, y, lambda_, mu, rho):
    if x == 0 and y == 0:
        return 1 - lambda_ * mu * rho
    elif x == 0 and y == 1:
        return 1 + lambda_ * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1

def neg_log_likelihood(params, data):
    alphas = params[:n_teams]
    betas = params[n_teams:2 * n_teams]
    gamma = params[-2]
    rho = params[-1]

    log_lik = 0

    for _, row in data.iterrows():
        i = team_idx[row['home_team_name']]
        j = team_idx[row['away_team_name']]

        lambda_ = alphas[i] * betas[j] * gamma
        mu = alphas[j] * betas[i]

        x = int(row['home_score'])
        y = int(row['away_score'])

        tau_ = tau(x, y, lambda_, mu, rho)
        if tau_ <= 0:
            return np.inf

        log_lik += (np.log(tau_) +
                    poisson.logpmf(x, lambda_) +
                    poisson.logpmf(y, mu))

    return -log_lik

# # Initial parameter guess
# x0 = np.concatenate([
#     np.ones(n_teams),   # alphas
#     np.ones(n_teams),   # betas
#     [1.3],              # gamma
#     [-0.1]              # rho
# ])

# # Constraints: keep alphas and betas positive, rho bounded
# bounds = ([(0.01, None)] * n_teams +   # alphas > 0
#           [(0.01, None)] * n_teams +   # betas > 0
#           [(1.0, 2.0)] +               # gamma
#           [(-1.0, 1.0)])               # rho

# print('Fitting model...')
# result = minimize(
#     neg_log_likelihood,
#     x0,
#     args=(home_train,),
#     method='L-BFGS-B',
#     bounds=bounds
# )

# # Extract fitted parameters
# alphas_fit = result.x[:n_teams]
# betas_fit = result.x[n_teams:2 * n_teams]
# gamma_fit = result.x[-2]
# rho_fit = result.x[-1]

# print(f'Optimisation success: {result.success}')
# print(f'Gamma (home advantage): {gamma_fit:.4f}')
# print(f'Rho (correction): {rho_fit:.4f}')

# Load parameters
params = np.load('dc_params.npy')
teams = np.load('dc_teams.npy', allow_pickle=True).tolist()
team_idx = {team: i for i, team in enumerate(teams)}
alphas_fit = params[:len(teams)]
betas_fit = params[len(teams):2*len(teams)]
gamma_fit = params[-2]
rho_fit = params[-1]

# Build one row per match for test set
home_test = test[test['team_id'] == test['home_team_id']].copy()
home_test['home_team_name'] = home_test['team_name']
home_test = home_test.merge(
    test[test['team_id'] == test['away_team_id']][['match_id', 'team_name']],
    on='match_id', suffixes=('', '_away')
)
home_test = home_test.rename(columns={'team_name_away': 'away_team_name'})


def predict_match(home_team, away_team, max_goals=10):
    i = team_idx[home_team]
    j = team_idx[away_team]

    lambda_ = alphas_fit[i] * betas_fit[j] * gamma_fit
    mu = alphas_fit[j] * betas_fit[i]

    score_matrix = np.zeros((max_goals, max_goals))
    for x in range(max_goals):
        for y in range(max_goals):
            score_matrix[x, y] = (tau(x, y, lambda_, mu, rho_fit) *
                                  poisson.pmf(x, lambda_) *
                                  poisson.pmf(y, mu))

    home_win = np.sum(np.tril(score_matrix, -1))
    draw = np.sum(np.diag(score_matrix))
    away_win = np.sum(np.triu(score_matrix, 1))

    most_likely = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win,
        'lambda': lambda_,
        'mu': mu,
        'pred_home_score': most_likely[0],
        'pred_away_score': most_likely[1]
    }


# Generate predictions for all test matches
predictions = []
for _, row in home_test.iterrows():
    pred = predict_match(row['home_team_name'], row['away_team_name'])
    pred['match_id'] = row['match_id']
    pred['home_team'] = row['home_team_name']
    pred['away_team'] = row['away_team_name']
    pred['actual_home_score'] = row['home_score']
    pred['actual_away_score'] = row['away_score']
    predictions.append(pred)

predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('dc_predictions.csv', sep=';', index=False)
print(predictions_df[['home_team', 'away_team', 'home_win', 'draw', 'away_win',
                       'pred_home_score', 'pred_away_score',
                       'actual_home_score', 'actual_away_score']].head(10))

def get_outcome(home_score, away_score):
    if home_score > away_score:
        return 'home_win'
    elif home_score == away_score:
        return 'draw'
    else:
        return 'away_win'

predictions_df['actual_outcome'] = predictions_df.apply(
    lambda r: get_outcome(r['actual_home_score'], r['actual_away_score']), axis=1
)
predictions_df['predicted_outcome'] = predictions_df[['home_win', 'draw', 'away_win']].idxmax(axis=1)

labels = ['home_win', 'draw', 'away_win']
cm = confusion_matrix(predictions_df['actual_outcome'], predictions_df['predicted_outcome'], labels=labels)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot()
plt.tight_layout()
plt.show()

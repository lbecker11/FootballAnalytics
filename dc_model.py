import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

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

# Initial parameter guess
x0 = np.concatenate([
    np.ones(n_teams),   # alphas
    np.ones(n_teams),   # betas
    [1.3],              # gamma
    [-0.1]              # rho
])

# Constraints: keep alphas and betas positive, rho bounded
bounds = ([(0.01, None)] * n_teams +   # alphas > 0
          [(0.01, None)] * n_teams +   # betas > 0
          [(1.0, 2.0)] +               # gamma
          [(-1.0, 1.0)])               # rho

print('Fitting model...')
result = minimize(
    neg_log_likelihood,
    x0,
    args=(home_train,),
    method='L-BFGS-B',
    bounds=bounds
)

# Extract fitted parameters
alphas_fit = result.x[:n_teams]
betas_fit = result.x[n_teams:2 * n_teams]
gamma_fit = result.x[-2]
rho_fit = result.x[-1]

print(f'Optimisation success: {result.success}')
print(f'Gamma (home advantage): {gamma_fit:.4f}')
print(f'Rho (correction): {rho_fit:.4f}')

# Save fitted parameters
np.save('dc_params.npy', result.x)
np.save('dc_teams.npy', np.array(teams))
print('Parameters saved to dc_params.npy and dc_teams.npy')

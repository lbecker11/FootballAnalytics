# Bundesliga Football Analytics

A full football analytics and betting edge pipeline built on five seasons of Bundesliga data (2020–2026). The project covers data collection, feature engineering, five separate prediction models, a stacking ensemble, a Kelly criterion backtest engine, and a Streamlit dashboard.

---

## Project Structure

```
footballanalytics/
├── data/
│   ├── bundesliga_features.csv      # Master feature file — one row per team per match
│   ├── bundesliga_odds.csv          # Pinnacle, Bet365, market average and max odds
│   ├── bundesliga_stats_all_seasons.csv
│   ├── dc_predictions.csv           # Dixon-Coles match probabilities
│   ├── xg_predictions.csv           # XGBoost xG regressor probabilities
│   ├── bp_predictions.csv           # Bivariate Poisson probabilities
│   ├── stacking_predictions.csv     # Stacking meta-learner probabilities
│   └── all_predictions.csv          # Combined — all model probs + Pinnacle odds
├── src/
│   ├── crawler.py                   # Scrapes match stats from the web
│   ├── load.py                      # Loads raw data into the database
│   ├── transform.py                 # Cleans and reshapes raw stats
│   ├── features.py                  # Feature engineering pipeline
│   ├── dc_model.py                  # Dixon-Coles model
│   ├── betting.py                   # XGBoost Classifier + backtest engine
│   ├── xg_model.py                  # XGBoost xG regressor → Poisson probabilities
│   ├── bivariate_poisson_model.py   # Bivariate Poisson model
│   ├── stacking_model.py            # Logistic Regression meta-learner
│   ├── bayesian_model.py            # Bayesian Network (explored, not used)
│   └── dashboard/
│       ├── app.py                   # Streamlit entry point
│       └── pages/
│           ├── 1_opportunity.py     # Per-match probability and edge breakdown
│           ├── 2_betting_decision.py
│           ├── 3_bet_tracker.py
│           ├── 4_season_summary.py
│           ├── 5_model_performance.py
│           └── 6_profitability.py
├── models/                          # Saved model artefacts
├── sql/                             # SQL feature queries
└── visualisations/                  # Saved confusion matrices and charts
```

---

## Methodology

### Data Collection

Match statistics were scraped for five Bundesliga seasons (2020–2026), covering goals, xG, shots on target, possession, passes, corners, fouls, and offsides from the official website. Odds data (Pinnacle, Bet365, market average and max) was sourced from football-data.co.uk.

### Feature Engineering

Features were built with strict time-awareness to prevent data leakage. Every feature at match time uses only information available before kick-off:

- **Previous match stats**: xG, shots, possession, accuracy from the prior game
- **Rolling averages**: goals, xG performance, shot conversion over recent matches
- **Form metrics**: cumulative points, win streak, points rank, home/away form windows
- **Head-to-head**: historical points from prior meetings
- **ELO ratings**: updated after every result, used as a strength proxy
- **Dixon-Coles λ/μ**: attack and defence rates from the fitted DC model, used as features in the XGBoost models
- **Matchday and home/away indicator**

All features were constructed per team per match, producing two rows per match (home and away perspective), which the models handle differently depending on their structure.

### Models

Five independent models were built, each using rolling window cross-validation — training on all seasons up to season *n*, predicting season *n+1*. This mirrors real-world deployment where future data is never available.

#### 1. Dixon-Coles

A maximum likelihood model that fits team attack (λ) and defence (μ) parameters from historical scorelines. Includes the Dixon-Coles low-score correction factor (ρ) to fix Poisson's tendency to underestimate draws at 0-0 and 1-1. Parameters are re-fitted each season fold. This model did not do the time adjustment, as my scraped data did not have dates. 

#### 2. XGBoost Classifier

A gradient boosted classifier predicting the three-way outcome (home win / draw / away win) directly from the engineered features. Trained with class balancing to address the natural imbalance between outcomes. Probability outputs are calibrated using isotonic regression fitted on a held-out validation season to correct the overconfident raw probabilities XGBoost produces.

#### 3. XGBoost xG (expected goals) Regressor

Sticking with the idea introduced by Dixon-Coles, this model predicts expected goals (λ) for each team separately. These lambdas are then fed into an independent Poisson score matrix with a Dixon-Coles ρ correction, from which home win / draw / away win probabilities are derived. This separates the scoring and outcome prediction steps.

#### 4. Bivariate Poisson

Extends the standard Poisson model by modelling home and away goals as correlated random variables via a shared component (λ3). Three XGBoost regressors predict λ1 (home goals), λ2 (away goals) and λ3 (covariance proxy). The full bivariate Poisson PMF is evaluated over a score matrix to produce outcome probabilities. Example:
| Home \ Away | 0     | 1     | 2     | 3     | 4     |
  |-------------|-------|-------|-------|-------|-------|
  | **0**       | 0.058 | 0.063 | 0.034 | 0.012 | 0.003 |
  | **1**       | 0.093 | 0.101 | 0.055 | 0.020 | 0.005 |
  | **2**       | 0.074 | 0.081 | 0.044 | 0.016 | 0.004 |
  | **3**       | 0.040 | 0.043 | 0.024 | 0.008 | 0.002 |
  | **4**       | 0.016 | 0.017 | 0.009 | 0.003 | 0.001 |

#### 5. Stacking (Meta-Learner)

A Logistic Regression meta-learner (softmax) trained on the probability outputs of all four models above. The meta-learner learns optimal weights for combining model outputs, with rolling window training to prevent leakage between seasons.

For each outcome class k in {home win, draw, away win}, the predicted probability is:

```
P(y = k) = exp(z_k) / (exp(z_home) + exp(z_draw) + exp(z_away))
```

Where z_k is the linear combination for class k:

```
z_k = b0 + w1*p_DC_home  + w2*p_DC_draw  + w3*p_DC_away
         + w4*p_XGB_home  + w5*p_XGB_draw + w6*p_XGB_away
         + w7*p_xG_home   + w8*p_xG_draw  + w9*p_xG_away
         + w10*p_BP_home  + w11*p_BP_draw + w12*p_BP_away
```

- b0 is the intercept for class k
- w1–w12 are the learned weights for each model's home, draw and away probability
- The softmax converts the raw scores into a probability distribution that sums to 1

There are 3 classes × 13 parameters (1 intercept + 12 weights) = 39 learnable parameters in total.

#### Bayesian Network (Explored, Not Used)

A discrete Bayesian Network was built using pgmpy with a hand-specified DAG connecting ELO, form, head-to-head, home advantage, and cumulative points through intermediate attack and defence strength nodes to the final outcome. Despite correct graph structure, the model consistently predicted home wins due to the coarseness of discretisation (bin issue) and limited sample size per CPT cell. The model ended up being too bias towards home wins. It was explored as a learning exercise but excluded from the final pipeline.

### Draw Threshold Optimisation

All probability models tend to underpredict draws because draws are the lowest-frequency outcome and the hardest to distinguish from close home/away wins in feature space. A draw threshold parameter was optimised for each model: if the predicted draw probability exceeds the threshold, the match is classified as a draw. The threshold is searched over [0.25, 0.40] subject to minimum draw recall (20%) and target accuracy (47%) constraints. This was deployed using a binary search. 

### Betting Edge Pipeline

The backtest engine (run from `betting.py`) applies the following steps for each of the five models:

1. **Vig removal**: Convert raw Pinnacle odds to fair implied probabilities by normalising the overround away. Pinnacle was chosen for its low margin (~2%) and reputation as the sharpest market.
2. **Edge**: `model_probability − fair_market_probability`
3. **Expected value (EV)**: `p × (odds − 1) − (1 − p)`
4. **Bet flagging**: Bets are placed where EV > 0.20 (home or away only — draw betting is excluded due to low model confidence). Away bets are additionally filtered at odds ≤ 10 to avoid extreme longshots where model reliability drops.
5. **Fractional Kelly sizing**: Stake = 25% Kelly fraction, capped at 10% of bankroll per bet. Fractional Kelly is used to reduce variance relative to full Kelly while preserving the positive expected growth property.
6. **Bankroll simulation**: Starting from €10,000, bets are placed chronologically across all seasons.

---

## Results

All accuracy figures are out-of-sample (rolling window, never trained on the season being predicted).

| Model | Accuracy | Draw Recall | Macro F1 |
|---|---|---|---|
| XGBoost Classifier | 0.481 | 0.28 | 0.447 |
| Dixon-Coles | 0.478 | 0.25 | 0.441 |
| XGBoost xG | 0.480 | 0.30 | 0.460 |
| Bivariate Poisson | 0.470 | 0.26 | 0.438 |
| Stacking | 0.470 | 0.40 | 0.453 |

The stacking model achieves the highest draw recall, at the cost of slightly lower overall accuracy. The XGBoost xG model achieves the best macro F1, suggesting more balanced class performance. No single model dominates across all metrics — which is typical in football prediction.

Bankroll simulation results varied by model and season. The betting pipeline is designed to demonstrate the methodology rather than claim profitability — five seasons is a small sample for drawing robust conclusions about edge. 

However, the stacked model performed the "best" by losing 42.8% of the bankroll. It also managed to accumulate €26'250 over 154 bets. It managed a 16.2% ROI in the 2022-2023 season with a final profit of €9293. 


---

## Limitations

- **No player-level data**: Injuries, suspensions, and lineup confirmations are unknown at prediction time and are one of the strongest signals in real betting markets.
- **Static odds**: The pipeline uses closing Pinnacle odds. Tracking line movement from opening to close (closing line value) would be a stronger validation of model quality and a signal in itself.
- **Small sample**: Five seasons (~1,700 matches) is sufficient to build the pipeline but limits the reliability of backtest conclusions. The stacking model in particular only has four training folds.
- **Model correlation**: All five models share the same underlying feature set, so the stacking ensemble gains are marginal. True ensemble diversity requires models with different information sources.
- **Draw exclusion**: Draws are excluded from the betting pipeline due to low model confidence. This limits opportunity and means the backtest does not capture the full picture.
- **Single league**: The pipeline was built specifically for the Bundesliga. Generalisability to other leagues with different playing styles or data availability is untested.
- **Dates**: Dates were missing from the data, which did not allow for fatigue calculations or the additional corrction in the Dixon-Coles paper. 

---

## What I Learned

### Scraping with Selenium
The stats site rendered its content dynamically via JavaScript, meaning a standard `requests` call returned an empty page. Selenium was needed to drive a real browser, wait for elements to load, and interact with dropdown menus to navigate between seasons and matchdays. The outline of the pages was always the same, which made it slightly easier. However, some rows were still misssed, but at such little quantity that it made no real impact on the model. 

### Data Wrangling
Standard pandas work. Handling missing values, reshaping between wide and long format, merging tables on keys. Nothing exotic but essential groundwork before anything else could be built.

### ELO Ratings
The ELO formula. Adapted from chess, rates team strength dynamically based on results. The core formula:

```
Expected = 1 / (1 + 10^((opponent_elo - team_elo) / 400))
New rating = Old rating + K × (Actual − Expected)
```

Where `Actual` is 1 for a win, 0.5 for a draw, 0 for a loss. `K` controls how quickly ratings update — K=40 for newly promoted teams and K=24 for established ones, since new teams have more uncertain ratings that should move faster. I saw this approach in another video about trying to systematically model tennis. This was a good choice as it was leading the SHAP values. 

### Putting a Research Paper into Practice
The Dixon-Coles model (Dixon & Coles, 1997) was the first time reading an academic paper and implementing it from scratch. The key insight was that standard Poisson models underestimate draws because goals aren't truly independent, the 0-0 and 1-1 scorelines are more common than Poisson predicts. Dixon and Coles introduced a correction factor ρ for these low-scoring outcomes that gets estimated from the data.

### Data Leakage
Mid-project it became clear that several features included stats from the current match (goals scored, shots taken, possession) which would be unknown at prediction time. A model trained on these features learns to fit the result rather than predict it, producing inflated accuracy on training data that collapses on unseen matches. Fixing this meant rebuilding all features to use only prior-match data (`prev_xgoals`, `rolling_avg_goals`, etc.).

### Probability Calibration — Isotonic Regression
XGBoost's raw probability outputs tend to be overconfident, pushing predictions toward 0 and 1 rather than reflecting true frequencies. `CalibratedClassifierCV` was explored first but removed in a newer sklearn version. The solution was fitting one `IsotonicRegression` per class on a held-out validation season, mapping raw scores to calibrated probabilities. Isotonic regression is non-parametric, it fits a monotone step function to the data rather than assuming a shape, making it well-suited to fixing arbitrary distortions in the raw outputs.

### Rolling window training
In order to have predicted outcomes for my data, so that the betting backtest could be created a new training approach was needed. Because XGBoost needed a validation set, it could not predict on that validation season. The idea was then to start with the 2020-2021 season, train on it and predict the 2021-2022 season. For the next iteration the trainining season was 2020-2021 again, we now have 2021-2022 as the validation and set and predict on 2022-2023. Then it would be 2020-2022 as training set, 2022-2023 as validation and then predict 2024-2025. This loop was done for all seasons. This way we had valid prediction data without any look ahead bias.  

### The Draw Problem and Threshold Optimisation
Draws are the hardest outcome to predict, they are the least frequent and the least distinguishable from close home or away wins in feature space. The initial models had near-zero draw recall, classifying almost everything as home or away win. The solution was a draw threshold: if the predicted draw probability exceeds a threshold, classify as draw. The threshold was found by searching 50 candidate values between 0.25 and 0.40, subject to minimum draw recall (20%) and minimum overall model accuracy (47%) constraints — a constrained optimisation over a discrete grid.
```
predicted = draw              if p_draw > threshold
predicted = argmax(p_h, p_d, p_a)   otherwise

subject to:
  accuracy  >= 0.47
  draw recall >= 0.20
```

Where threshold is searched over 50 values between 0.25 and 0.40, and argmax is the standard winner-takes-all prediction when draw is not triggered.

### Accuracy vs. Recall
Overall accuracy masks imbalanced performance across classes. A model that predicts home win for every match achieves ~45% accuracy in the Bundesliga — without correctly identifying a single draw or away win. Recall per class — the fraction of actual draws, home wins, and away wins correctly identified — is what matters for understanding where the model actually works. The two metrics need to be read together: accuracy tells you how often you are right overall, recall tells you which outcomes you are actually capturing. This was essential since our main test was testing it with a simulated bankroll. Thats why we used recall as constraint in the draw problem optimisation. 

### How Odds Work and Converting to Probabilities
Bookmaker odds imply a probability — a Pinnacle line of 2.10 on a home win implies the market thinks there is a 1/2.10 = 47.6% chance of it happening. But if you sum the implied probabilities across all three outcomes they add up to more than 100% — typically 102–106% depending on the bookmaker. That excess is the vig (or overround) — the bookmaker's built-in margin that ensures profit regardless of outcome.

To get fair probabilities you remove the vig by normalising:

```
fair_home = (1/odds_home) / ((1/odds_home) + (1/odds_draw) + (1/odds_away))
```

This converts raw odds into a proper probability distribution that sums to 1. Pinnacle was used specifically because their margin is ~2% (the lowest in the industry) making their fair probabilities as close to the true market consensus as publicly available data allows.

This same logic applies anywhere odds exist. In poker, pot odds work identically. If you need to call €10 into a €30 pot you are getting 3:1, implying you need to win 25% of the time to break even. Converting between odds and probabilities is a transferable skill across any domain with uncertain outcomes.

### Edge and Expected Value
Edge is the gap between what the model thinks the probability is and what the market implies:

```
edge = model_probability − fair_market_probability
```

A positive edge means the model believes the outcome is more likely than the market is pricing. Expected value quantifies how much you expect to win or lose per unit staked:

```
EV = p × (odds − 1) − (1 − p)
```

A positive EV bet is one where, averaged over many repetitions, you come out ahead. A single positive EV bet can still lose, EV is a long-run concept. We filtered bets to EV > 0.20 to require a meaningful edge rather than betting on marginal situations.

### Kelly Criterion
Kelly answers the question of how much to stake given a positive EV bet:

```
f = (b × p − (1 − p)) / b
```

Where `b` is the decimal odds minus 1 and `p` is the model's probability. Full Kelly maximises long-run bankroll growth but produces very large stakes and high variance. Fractional Kelly (25% of the full Kelly stake) capped at 10% of bankroll per bet was used, a standard practitioner adjustment that sacrifices some expected growth in exchange for significantly smoother bankroll development and protection against model error. Kelly inherently scales stakes with confidence, a high-edge bet gets a larger stake automatically, without any manual judgement.

The broader insight connecting prediction and betting: a model that is right 48% of the time can still have positive EV if it is right on outcomes the market underprices. Prediction accuracy and betting profitability are related but not the same thing. This same logic can be applied to deep value investing. As long as the winners are high enough they can cover the losers plus profit. 

---

## Running the Project

**Dashboard:**
```bash
streamlit run src/dashboard/app.py
```

**Backtest engine (all 5 models, comparison chart):**
```bash
python src/betting.py
```

**Individual models** (each saves predictions to `data/`):
```bash
python src/dc_model.py
python src/xg_model.py
python src/bivariate_poisson_model.py
python src/stacking_model.py
```

---

## Dependencies

Python 3.13. Key libraries: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `scipy`, `streamlit`, `plotly`, `pgmpy`, `matplotlib`.

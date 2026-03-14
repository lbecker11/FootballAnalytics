# Date Cleaning
import pandas as pd 
import numpy as np 
from typing import List, Any

df = pd.read_csv('bundesliga_stats_all_seasons.csv', sep=",")

df_c = df.copy()
# type casting 
df_c['MatchDay'] = df_c['MatchDay'].replace({'Relegation': '35'}).astype(int)

# Split relegation two-legged ties into matchday 35 and 36
for season in df_c['Season'].unique(): #type: ignore
    md35 = df_c[(df_c['Season'] == season) & (df_c['MatchDay'] == 35)]
    if len(md35) == 4:
        df_c.loc[md35.index[2:], 'MatchDay'] = 36

# Remove none played games for later
df_matches_played = df_c[df_c[['Scored', 'Shots', 'Possession']].notna().all(axis=1)]

# Check for nulls and remove
def _null_check() -> List[Any]:
    games_to_check: List[Any] = []
    for i, game in df_matches_played.iterrows():
        if game.isnull().any():
            games_to_check.append(i)
    return games_to_check 
# Logical checks
def drop_empties() -> None:
    return df_matches_played.drop(index=_null_check(), inplace=True)
def _check_shots():
    return df_matches_played[df_matches_played['Shots'] != df_matches_played['OnTarget'] + df_matches_played['OffTarget']]
def _check_wins():
    return df_matches_played[(df_matches_played['Scored'] > df_matches_played['Conceded'])&(df_matches_played['Win'] != 1)]
def _check_possession():
    return df_matches_played[(df_matches_played['Possession'] <= 0) | (df_matches_played['Possession'] >= 1)]
def _check_team_names():
    all_teams = set()
    for season in df_matches_played['Season'].unique(): #type: ignore
        teams = set(df_matches_played.loc[df_matches_played['Season'] == season, 'Team'].unique())
        all_teams = all_teams.union(teams)
    assert len(all_teams) <= 30, "Too many teams"
def _check_home_away():
    assert list(df_matches_played['HomeAway'].unique()) == ['Home', 'Away'], "Wrong Home/Away values" #type: ignore
def _check_matchday_range():
    matchdays = sorted(df_matches_played['MatchDay'].unique()) #type: ignore
    valid_matchdays = list(range(1, 35)) + [35, 36]
    for md in matchdays:
        assert md in valid_matchdays, f"Matchday {md} out of range" #type: ignore
def _check_scored():
    own_goals = df_matches_played[df_matches_played['Scored'] > df_matches_played['OnTarget']]
    if len(own_goals) > 0:
        print(f"Warning: {len(own_goals)} matches with likely own goals (Scored > OnTarget)")
    return own_goals
def _check_accuracy():
    return df_matches_played[(df_matches_played['Accuracy'] <= 0) | (df_matches_played['Accuracy'] >= 1)]
def _check_duplicates():
    regular_season = df_matches_played[~df_matches_played['MatchDay'].isin([35, 36])] #type: ignore
    return regular_season[regular_season.duplicated(subset=['Season', 'MatchDay', 'Team'], keep=False)] #type: ignore
def _check_symmetry():
    mismatches = []
    for i, row in df_matches_played.iterrows():
        opponent_row = df_matches_played[
            (df_matches_played['Season'] == row['Season']) &
            (df_matches_played['MatchDay'] == row['MatchDay']) &
            (df_matches_played['Team'] == row['Opponent']) &
            (df_matches_played['Opponent'] == row['Team'])
        ]
        if opponent_row.empty: #type: ignore
            mismatches.append(i)
        elif opponent_row.iloc[0]['Scored'] != row['Conceded']: #type: ignore
            mismatches.append(i)
    return df_matches_played.loc[mismatches]
def get_clean_data():
    drop_empties()
    assert len(_check_shots()) == 0, "Shot totals are not adding up"
    assert len(_check_wins()) == 0, "Win flags are inconsistent"
    assert len(_check_possession()) == 0, "Possession out of range"
    assert len(_check_accuracy()) == 0, "Accuracy out of range"
    _check_scored()
    assert len(_check_duplicates()) == 0, "Duplicate rows found"
    assert len(_check_symmetry()) == 0, "Bad symmetry"
    _check_team_names()
    _check_home_away()
    _check_matchday_range()

    return df_matches_played


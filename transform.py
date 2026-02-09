# Date Cleaning
import pandas as pd 
import numpy as np 
from typing import List, Any

df = pd.read_csv('bundesliga_stats_all_seasons.csv', sep=",")

df_c = df.copy()
# type casting 
df_c['MatchDay'] = df_c['MatchDay'].replace({'Relegation': '35'}).astype(int)

# Remove none played games for later
df_matches_played = df_c[df_c[['Scored', 'Shots', 'Possession']].notna().all(axis=1)]

# Check for nulls and remove
def _null_check() -> List[Any]:
    games_to_check: List[Any] = []
    for i, game in df_matches_played.iterrows():
        if game.isnull().any():
            games_to_check.append(i)
    return games_to_check 
def _drop_empties() -> None:
    df_matches_played.drop(index=_null_check(), inplace=True)
#TODO: Shots = OnTarget + OffTarget; Win = 1 if Scored > Conceded; Scored <= OnTarget; 0<Possession<1 
def _check_shots():
    return df_matches_played[df_matches_played['Shots'] != df_matches_played['OnTarget'] + df_matches_played['OffTarget']]
def _check_wins():
    return df_matches_played[(df_matches_played['Scored'] > df_matches_played['Conceded'])&(df_matches_played['Win'] != 1)]


 





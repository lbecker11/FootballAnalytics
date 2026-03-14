import psycopg2
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
PASSWORD = os.getenv("PASSWORD")

conn = psycopg2.connect(
    dbname="football_analytics",
    user="postgres",
    password=PASSWORD,
    host="localhost",
    port="5432"
)

def rebuild_features_table():
    with open('../sql/feature_query.txt', 'r') as f:
        sql = f.read()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS features;")
    cur.execute(sql)
    conn.commit()
    cur.close()
    print("Features table rebuilt successfully")

rebuild_features_table()

query = """
    SELECT ms.match_id, ms.team_id, ms.scored, ms.conceded, ms.xgoals, ms.passes,
    ms.accuracy, ms.shots, ms.ontarget, ms.offtarget, ms.possession, ms.tackleswon,
    ms.corners, ms.offsides, ms.fouls, ms.win,
    CASE WHEN ms.team_id = m.home_team_id THEN 'Home' ELSE 'Away' END as homeaway,
    f.rolling_avg_goals, f.prev_xgoals, f.prev_possession, f.prev_shots,
    f.prev_ontarget, f.prev_accuracy, f.cumulative_points, f.points_rank,
    f.goal_difference, f.win_streak,
    m.matchday, m.home_team_id, m.away_team_id, m.home_score, m.away_score,
    s.season_name, t.team_name
    FROM match_stats ms
    JOIN features f ON ms.match_id = f.match_id AND ms.team_id = f.team_id
    JOIN matches m ON ms.match_id = m.match_id
    JOIN seasons s ON m.season_id = s.season_id
    JOIN teams t ON ms.team_id = t.team_id """

df = pd.read_sql(query, conn)

def calculate_elo(k_new=40, k_established=24):
    match_query = """
        SELECT m.match_id, m.home_team_id, m.away_team_id, m.home_score, m.away_score,
        m.matchday, s.season_name
        FROM matches m
        JOIN seasons s ON m.season_id = s.season_id
        """
    df_matches = pd.read_sql(match_query, conn)
    df_matches = df_matches.sort_values(['season_name', 'matchday'])

    elo_ratings = {}
    seasons_played = {}
    results = []

    for season, season_df in df_matches.groupby('season_name', sort=True):
        # Track seasons in league for each team in this season
        season_teams = set(
            season_df['home_team_id'].tolist() + season_df['away_team_id'].tolist()
        )
        for team in season_teams:
            seasons_played[team] = seasons_played.get(team, 0) + 1

        for _, match in season_df.sort_values('matchday').iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']

            # Init 1500 rating for first occurrence
            if home_id not in elo_ratings:
                elo_ratings[home_id] = 1500
            if away_id not in elo_ratings:
                elo_ratings[away_id] = 1500

            home_elo = elo_ratings[home_id]
            away_elo = elo_ratings[away_id]

            # Assign K based on seasons in league
            k_home = k_new if seasons_played.get(home_id, 1) == 1 else k_established
            k_away = k_new if seasons_played.get(away_id, 1) == 1 else k_established

            # Expected result
            home_expected = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            away_expected = 1 - home_expected

            # Actual result
            if match['home_score'] > match['away_score']:
                home_actual, away_actual = 1, 0
            elif match['home_score'] == match['away_score']:
                home_actual, away_actual = 0.5, 0.5
            else:
                home_actual, away_actual = 0, 1

            # Update ratings
            home_new = home_elo + k_home * (home_actual - home_expected)
            away_new = away_elo + k_away * (away_actual - away_expected)

            results.append({
                'match_id': match['match_id'],
                'home_team_id': home_id,
                'away_team_id': away_id,
                'home_elo_before': home_elo,
                'away_elo_before': away_elo,
                'home_elo_after': home_new,
                'away_elo_after': away_new
            })

            elo_ratings[home_id] = home_new
            elo_ratings[away_id] = away_new

    return pd.DataFrame(results)

def calculate_h2h(df):
    df = df.sort_values(['season_name', 'matchday']).copy()
    df['opponent_id'] = np.where(
        df['team_id'] == df['home_team_id'],
        df['away_team_id'],
        df['home_team_id']
    )
    df['points'] = np.where(df['win'] == True, 3,
                   np.where(df['scored'] == df['conceded'], 1, 0))
    df['h2h_points'] = 0

    for (team, opp), group in df.groupby(['team_id', 'opponent_id']):
        cumulative = group['points'].shift(1).expanding().sum().fillna(0)
        df.loc[group.index, 'h2h_points'] = cumulative.values

    df = df.drop(columns=['points'])
    return df

def calculate_home_away_form(df):
    df = df.sort_values(['season_name', 'matchday']).copy()

    home_df = df[df['homeaway'] == 'Home'].copy()
    away_df = df[df['homeaway'] == 'Away'].copy()

    df['home_form_wins'] = (home_df.groupby('team_id')['win']
                            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()))

    df['home_form_goals'] = (home_df.groupby('team_id')['scored']
                             .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

    df['away_form_wins'] = (away_df.groupby('team_id')['win']
                            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()))

    df['away_form_goals'] = (away_df.groupby('team_id')['scored']
                             .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))

    return df

def calculate_xg_performance(df):
    df = df.sort_values(['season_name', 'matchday']).copy()
    xg_performance = df['scored'] - df['xgoals']
    df['rolling_xg_performance'] = (df.groupby('team_id')[df.columns[0]]
                                    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    df['rolling_xg_performance'] = (df.assign(xg_performance=xg_performance)
                                    .groupby('team_id')['xg_performance']
                                    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    return df

def calculate_shot_conversion(df):
    df = df.sort_values(['season_name', 'matchday']).copy()
    shot_conversion = df['scored'] / df['shots'].replace(0, float('nan'))
    df['rolling_shot_conversion'] = (df.assign(shot_conversion=shot_conversion)
                                     .groupby('team_id')['shot_conversion']
                                     .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()))
    return df

def build_features():
    # Apply all feature functions
    result = calculate_xg_performance(df)
    result = calculate_home_away_form(result)
    result = calculate_shot_conversion(result)
    result = calculate_h2h(result)

    # Join ELO — one row per match, need to map to team perspective
    elo_df = calculate_elo()
    result = result.merge(elo_df[['match_id', 'home_team_id', 'away_team_id',
                                   'home_elo_before', 'away_elo_before']],
                          on='match_id', suffixes=('', '_elo'))
    result['elo_before'] = np.where(
        result['team_id'] == result['home_team_id_elo'],
        result['home_elo_before'],
        result['away_elo_before']
    )
    result['opponent_elo_before'] = np.where(
        result['team_id'] == result['home_team_id_elo'],
        result['away_elo_before'],
        result['home_elo_before']
    )
    result = result.drop(columns=['home_team_id_elo', 'away_team_id_elo',
                                   'home_elo_before', 'away_elo_before'])

    result.to_csv('../data/bundesliga_features.csv', index=False)
    print(f'Exported to bundesliga_features.csv — {len(result)} rows, {len(result.columns)} columns')

    return result


if __name__ == '__main__':
    build_features()



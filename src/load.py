import psycopg2
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
from dotenv import load_dotenv
import os
from transform import get_clean_data

register_adapter(np.int64, lambda val: AsIs(int(val)))
register_adapter(np.float64, lambda val: AsIs(float(val)))
load_dotenv()

conn = psycopg2.connect(
    dbname="football_analytics",
    user="postgres",
    password=os.getenv('PASSWORD'),
    host="localhost",
    port="5432"
)

curr = conn.cursor()

df = get_clean_data()
# INSERT INTO tables 
curr.executemany("INSERT INTO teams (team_name) VALUES (%s)", [(t,) for t in list(df['Team'].unique())]) #teams
curr.executemany("INSERT INTO seasons (season_name) VALUES (%s)", [(s,) for s in list(df['Season'].unique())]) #seasons
curr.execute("SELECT team_name, team_id FROM teams")
team_map = dict(curr.fetchall())
curr.execute("SELECT season_name, season_id FROM seasons")
season_map = dict(curr.fetchall())
home_rows = df[df['HomeAway'] == 'Home']
for i, row in home_rows.iterrows():
    # Insert match and get match id
    curr.execute("""
    INSERT INTO matches (season_id, matchday, home_team_id, away_team_id, home_score,
    away_score) VALUES (%s, %s, %s, %s, %s, %s) RETURNING match_id""",
                 (season_map[row['Season']], row['MatchDay'], team_map[row['Team']],
                  team_map[row['Opponent']], row['Scored'], row['Conceded']))
    
    match_id = curr.fetchone()[0]
    
    # Home stats 
    curr.execute("""
    INSERT INTO match_stats (match_id, team_id, scored, conceded, xgoals, passes, accuracy,
    shots, ontarget, offtarget, possession, tackleswon, corners, offsides, fouls, win)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                 (match_id, team_map[row['Team']], row['Scored'], row['Conceded'], row['XGoals'],
                  row['Passes'], row['Accuracy'], row['Shots'], row['OnTarget'], row['OffTarget'],
                  row['Possession'], row['TacklesWon'], row['Corners'], row['Offsides'],
                  row['Fouls'], bool(row['Win'])))

    # Away Stats 
    away_row = df[(df['Season'] == row['Season']) &
                 (df['MatchDay'] == row['MatchDay']) &
                 (df['Team'] == row['Opponent']) & 
                 (df['HomeAway'] == 'Away')].iloc[0]
    curr.execute("""
    INSERT INTO match_stats (match_id, team_id, scored, conceded, xgoals, passes, accuracy,
    shots, ontarget, offtarget, possession, tackleswon, corners, offsides, fouls, win) 
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                 (match_id, team_map[away_row['Team']], away_row['Scored'], away_row['Conceded'],
                  away_row['XGoals'], away_row['Passes'], away_row['Accuracy'], away_row['Shots'],
                  away_row['OnTarget'], away_row['OffTarget'], away_row['Possession'],
                  away_row['TacklesWon'], away_row['Corners'], away_row['Offsides'],
                  away_row['Fouls'], bool(away_row['Win'])))
conn.commit() 

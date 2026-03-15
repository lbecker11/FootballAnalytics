import requests
import pandas as pd
import os

# Resolve paths relative to this file's location so script works from any directory
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')

BASE_URL = 'https://www.football-data.co.uk/'

SEASONS = {
    '2020-2021': 'mmz4281/2021/D1.csv',
    '2021-2022': 'mmz4281/2122/D1.csv',
    '2022-2023': 'mmz4281/2223/D1.csv',
    '2023-2024': 'mmz4281/2324/D1.csv',
    '2024-2025': 'mmz4281/2425/D1.csv',
    '2025-2026': 'mmz4281/2526/D1.csv',
}

def download_odds():
    dfs = []
    for season, path in SEASONS.items():
        url = BASE_URL + path
        response = requests.get(url)
        response.raise_for_status()

        # Parse CSV from response content
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df['season_name'] = season
        dfs.append(df)
        print(f'Downloaded {season} — {len(df)} matches')

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(DATA_DIR, 'bundesliga_odds.csv'), index=False)
    print(f'\nSaved bundesliga_odds.csv — {len(combined)} total matches')
    return combined

if __name__ == '__main__':
    download_odds()

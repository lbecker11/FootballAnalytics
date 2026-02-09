from typing import List, Dict, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

class BundesligaCrawler():
    SEASONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025', '2025-2026']
    MATCHDAYS = list(range(1, 35)) + ['Relegation']  # 1-34 + Relegation

    def __init__(self, year: str = None, matchday: int = None, url: str = 'https://www.bundesliga.com/en/bundesliga/matchday/'):
        self.url = url
        self.year = year
        self.matchday = matchday
        self.base_url = f"{url}{year}/{matchday}" if year and matchday else None
        self.all_stats: List[Dict] = []
        
    def setup_driver(self) -> webdriver.Chrome:
        """Initialize Chrome driver with options"""
        options = Options()
        # options.add_argument('--headless')  # Commented out for debugging
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        driver = webdriver.Chrome(options=options)
        return driver
    
    def get_match_links(self, driver: webdriver.Chrome) -> List[str]:
        """Get all match fixture links from the matchday page"""
        print(f"Loading matchday page: {self.base_url}")
        driver.get(self.base_url)
        
        # Wait for match fixtures to load
        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'matchFixture'))
        )
        
        # Find all match fixtures
        fixtures = driver.find_elements(By.CLASS_NAME, 'matchFixture')
        print(f"Found {len(fixtures)} fixtures")
        
        match_links = []
        for i, fixture in enumerate(fixtures):
            href = fixture.get_attribute('href')
            
            if href:
                # Check if it's a relative URL
                if href.startswith('/'):
                    full_url = f"https://www.bundesliga.com{href}"
                else:
                    full_url = href
                
                # Replace 'liveticker' with 'stats'
                if 'liveticker' in full_url:
                    stats_link = full_url.replace('liveticker', 'stats')
                else:
                    stats_link = full_url + '/stats' if not full_url.endswith('/') else full_url + 'stats'
                
                match_links.append(stats_link)
                print(f"Match {i+1} stats: {stats_link}")
        
        return match_links
    
    def _find_bar_values(self, soup: BeautifulSoup, stat_title: str) -> Tuple[Optional[str], Optional[str]]:
        """Find left/right bar values for a given stat title"""
        # Try stats-section first
        for section in soup.find_all(class_='stats-section'):
            title_elem = section.find(class_='title')
            if title_elem and stat_title.lower() in title_elem.get_text().lower():
                left = section.find(class_=lambda x: x and 'value' in x and 'left' in x)
                right = section.find(class_=lambda x: x and 'value' in x and 'right' in x)
                if left and right:
                    return left.get_text(strip=True), right.get_text(strip=True)
        return None, None

    def _find_bar_values_by_header(self, content: str, stat_title: str) -> Tuple[Optional[str], Optional[str]]:
        """Find bar values by searching for h2/h4 header followed by bar-chart"""
        pattern = rf'{stat_title}</h[24]>.*?class="value left"[^>]*>\s*(\d+).*?class="value right"[^>]*>\s*(\d+)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _find_sub_stat_values(self, content: str, stat_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Find values for sub-stats like Offsides, Fouls committed"""
        idx = content.find(stat_name)
        if idx > 0:
            region = content[idx:idx+700]
            vals = re.findall(r'value[^>]*>\s*(\d+)', region)
            if len(vals) >= 2:
                return vals[0], vals[1]
        return None, None

    def extract_stats_from_page(self, driver: webdriver.Chrome) -> Optional[List[Dict]]:
        """Extract statistics from the current stats page and return two rows (home/away)"""
        try:
            time.sleep(3)
            print(f"\nExtracting stats from: {driver.current_url}")

            content = driver.page_source
            soup = BeautifulSoup(content, 'html.parser')

            # Extract team names from page title
            title_match = re.search(r'<title>([^<]+)</title>', content)
            home_team, away_team = None, None
            if title_match:
                teams = re.findall(r'([A-Za-zÄÖÜäöüß\s\d]+?)\s*-\s*([A-Za-zÄÖÜäöüß\s\d]+?)\s*\|', title_match.group(1))
                if teams:
                    home_team = teams[0][0].strip()
                    away_team = teams[0][1].strip()
            print(f"  Teams: {home_team} vs {away_team}")

            # Extract score
            scores = soup.find_all(class_='matchcenter-score-card-value')
            home_score, away_score = 0, 0
            if len(scores) >= 2:
                home_score = int(scores[0].get_text(strip=True))
                away_score = int(scores[1].get_text(strip=True))
            print(f"  Score: {home_score} - {away_score}")

            # Extract xGoals (decimal values in xGoals section)
            home_xg, away_xg = None, None
            for section in soup.find_all(class_='stats-section'):
                title = section.find(class_='title')
                if title and 'xGoals' in title.get_text():
                    all_vals = section.find_all(class_=re.compile('value'))
                    decimal_vals = [v.get_text(strip=True) for v in all_vals
                                    if '.' in v.get_text(strip=True)]
                    if len(decimal_vals) >= 2:
                        home_xg, away_xg = decimal_vals[0], decimal_vals[1]
            print(f"  xGoals: {home_xg} vs {away_xg}")

            # Extract Passes
            home_passes, away_passes = self._find_bar_values(soup, 'Passes completed')
            print(f"  Passes: {home_passes} vs {away_passes}")

            # Extract Accuracy (from text-chart within Passes section)
            home_acc, away_acc = None, None
            idx = content.find('Passes completed')
            if idx > 0:
                region = content[idx:idx+3000]
                left_match = re.search(r'left[^>]*>(\d+)\s*%', region, re.IGNORECASE)
                right_match = re.search(r'right[^>]*>(\d+)\s*%', region, re.IGNORECASE)
                if left_match and right_match:
                    home_acc = left_match.group(1)
                    away_acc = right_match.group(1)
            print(f"  Accuracy: {home_acc}% vs {away_acc}%")

            # Extract Shots (on target and off target)
            home_on, home_off, away_on, away_off = 0, 0, 0, 0
            for section in soup.find_all(class_='stats-section'):
                title = section.find(class_='title')
                if title and title.get_text(strip=True) == 'Shots':
                    all_text = section.get_text()
                    on_targets = re.findall(r'(\d+)\s*on\s*target', all_text, re.IGNORECASE)
                    off_targets = re.findall(r'(\d+)\s*off\s*target', all_text, re.IGNORECASE)
                    if len(on_targets) >= 2:
                        home_on, away_on = int(on_targets[0]), int(on_targets[1])
                    if len(off_targets) >= 2:
                        home_off, away_off = int(off_targets[0]), int(off_targets[1])
            home_shots = home_on + home_off
            away_shots = away_on + away_off
            print(f"  Shots: {home_shots} ({home_on}/{home_off}) vs {away_shots} ({away_on}/{away_off})")

            # Extract Possession (look for 2-digit percentages)
            home_poss, away_poss = None, None
            poss_pct = re.findall(r'>(\d{2})<', content)
            poss_candidates = [int(p) for p in poss_pct if 20 <= int(p) <= 80]
            if len(poss_candidates) >= 2:
                home_poss, away_poss = poss_candidates[0], poss_candidates[1]
            print(f"  Possession: {home_poss}% vs {away_poss}%")

            # Extract Tackles won (uses h2 header, not stats-section title)
            home_tackles, away_tackles = self._find_bar_values_by_header(content, 'Tackles won')
            print(f"  Tackles: {home_tackles} vs {away_tackles}")

            # Extract Corners
            home_corners, away_corners = self._find_bar_values(soup, 'Corners')
            print(f"  Corners: {home_corners} vs {away_corners}")

            # Extract Offsides (sub-stat)
            home_offsides, away_offsides = self._find_sub_stat_values(content, '>Offsides<')
            print(f"  Offsides: {home_offsides} vs {away_offsides}")

            # Extract Fouls committed (sub-stat)
            home_fouls, away_fouls = self._find_sub_stat_values(content, 'Fouls committed')
            print(f"  Fouls: {home_fouls} vs {away_fouls}")

            # Calculate win (1 if scored > conceded, 0 otherwise)
            home_win = 1 if home_score > away_score else 0
            away_win = 1 if away_score > home_score else 0

            # Convert accuracy to decimal (90 -> 0.90)
            home_acc_decimal = round(int(home_acc) / 100, 2) if home_acc else None
            away_acc_decimal = round(int(away_acc) / 100, 2) if away_acc else None

            # Convert possession to decimal (61 -> 0.61)
            home_poss_decimal = round(home_poss / 100, 2) if home_poss else None
            away_poss_decimal = round(away_poss / 100, 2) if away_poss else None

            # Create two rows: one for home team, one for away team
            home_row = {
                'Season': self.year,
                'MatchDay': self.matchday,
                'Team': home_team,
                'HomeAway': 'Home',
                'Opponent': away_team,
                'Scored': home_score,
                'Conceded': away_score,
                'XGoals': home_xg,
                'Passes': home_passes,
                'Accuracy': home_acc_decimal,
                'Shots': home_shots,
                'OnTarget': home_on,
                'OffTarget': home_off,
                'Possession': home_poss_decimal,
                'TacklesWon': home_tackles,
                'Corners': home_corners,
                'Offsides': home_offsides,
                'Fouls': home_fouls,
                'Win': home_win
            }

            away_row = {
                'Season': self.year,
                'MatchDay': self.matchday,
                'Team': away_team,
                'HomeAway': 'Away',
                'Opponent': home_team,
                'Scored': away_score,
                'Conceded': home_score,
                'XGoals': away_xg,
                'Passes': away_passes,
                'Accuracy': away_acc_decimal,
                'Shots': away_shots,
                'OnTarget': away_on,
                'OffTarget': away_off,
                'Possession': away_poss_decimal,
                'TacklesWon': away_tackles,
                'Corners': away_corners,
                'Offsides': away_offsides,
                'Fouls': away_fouls,
                'Win': away_win
            }

            return [home_row, away_row]

        except Exception as e:
            print(f"Error extracting stats: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def scrape_all_matches(self, driver: webdriver.Chrome = None) -> pd.DataFrame:
        """Scrape stats from all matches on the matchday"""
        own_driver = driver is None
        if own_driver:
            driver = self.setup_driver()

        try:
            match_links = self.get_match_links(driver)

            if not match_links:
                print("No match links found!")
                return pd.DataFrame()

            for i, link in enumerate(match_links):
                print(f"\n{'='*60}")
                print(f"Processing match {i+1}/{len(match_links)}")
                print(f"{'='*60}")

                driver.get(link)
                time.sleep(3)

                stats_rows = self.extract_stats_from_page(driver)
                if stats_rows:
                    self.all_stats.extend(stats_rows)

            df = pd.DataFrame(self.all_stats)
            print(f"\n{'='*60}")
            print(f"Scraped {len(df)} team rows from {len(match_links)} matches")
            print(f"{'='*60}")

            return df

        except Exception as e:
            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if own_driver:
                driver.quit()

        return pd.DataFrame()

    def scrape_all_seasons(self, seasons: List[str] = None, matchdays: List = None) -> pd.DataFrame:
        """Scrape stats from all seasons and matchdays"""
        seasons = seasons or self.SEASONS
        matchdays = matchdays or self.MATCHDAYS

        driver = self.setup_driver()
        all_data: List[Dict] = []

        try:
            total_matchdays = len(seasons) * len(matchdays)
            current = 0

            for season in seasons:
                for matchday in matchdays:
                    current += 1
                    matchday_label = matchday if isinstance(matchday, str) else str(matchday)

                    print(f"\n{'#'*60}")
                    print(f"# Season {season} - Matchday {matchday_label} ({current}/{total_matchdays})")
                    print(f"{'#'*60}")

                    # Update instance variables for this matchday
                    self.year = season
                    self.matchday = matchday
                    self.base_url = f"{self.url}{season}/{matchday_label}"
                    self.all_stats = []

                    try:
                        match_links = self.get_match_links(driver)

                        if not match_links:
                            print(f"No matches found for {season} matchday {matchday_label}, skipping...")
                            continue

                        for i, link in enumerate(match_links):
                            print(f"  Match {i+1}/{len(match_links)}")
                            driver.get(link)
                            time.sleep(2)

                            stats_rows = self.extract_stats_from_page(driver)
                            if stats_rows:
                                all_data.extend(stats_rows)

                        print(f"  Completed: {len(self.all_stats)} rows collected")

                    except Exception as e:
                        print(f"  Error on {season} matchday {matchday_label}: {e}")
                        continue

            df = pd.DataFrame(all_data)
            print(f"\n{'#'*60}")
            print(f"# TOTAL: Scraped {len(df)} team rows")
            print(f"{'#'*60}")

            return df

        except Exception as e:
            print(f"\nFatal error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            driver.quit()

        return pd.DataFrame(all_data) if all_data else pd.DataFrame()

# Usage
if __name__ == "__main__":
    import sys

    crawler = BundesligaCrawler()

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Scrape all seasons and matchdays
        print("Scraping ALL seasons and matchdays...")
        df = crawler.scrape_all_seasons()
        if not df.empty:
            filename = 'bundesliga_stats_all_seasons.csv'
            df.to_csv(filename, index=False)
            print(f"\nSaved {len(df)} rows to {filename}")
    else:
        # Default: scrape single matchday for testing
        crawler = BundesligaCrawler(year='2020-2021', matchday=1)
        df = crawler.scrape_all_matches()
        if not df.empty:
            print("\nDataFrame columns:", df.columns.tolist())
            print("\nSample data:")
            print(df.to_string())
            df.to_csv(f'bundesliga_stats_{crawler.year}_md{crawler.matchday}.csv', index=False)
            print(f"\nSaved to bundesliga_stats_{crawler.year}_md{crawler.matchday}.csv")

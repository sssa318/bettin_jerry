# 01_data_collection.py

import pandas as pd
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from nba_api.stats.static import players, teams
import os
import time

def get_player_id(player_name):
    """Fetches a player's ID from their full name."""
    try:
        player_dict = players.find_players_by_full_name(player_name)
        if player_dict:
            return player_dict[0]['id']
        else:
            return None
    except Exception as e:
        print(f"Error finding player ID for {player_name}: {e}")
        return None

def fetch_player_gamelogs(player_list, season='2023-24', num_games=82):
    """Fetches recent game logs for a list of players."""
    all_gamelogs = pd.DataFrame()
    for player_name in player_list:
        player_id = get_player_id(player_name)
        if not player_id:
            print(f"❌ Skipping {player_name}: Player ID not found.")
            continue
        
        try:
            # Add a small delay to avoid overwhelming the API
            time.sleep(1)
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season')
            df = gamelog.get_data_frames()[0]
            
            df = df.head(num_games)
            df['PLAYER_NAME'] = player_name

            # Extract opponent and home/away status
            df['HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
            df['OPPONENT_ABBREVIATION'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

            # Keep only the columns we need for now
            expected_cols = ['PLAYER_NAME', 'GAME_DATE', 'OPPONENT_ABBREVIATION', 'HOME',
                               'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS']
            
            keep_cols = [col for col in expected_cols if col in df.columns]
            df = df[keep_cols]

            all_gamelogs = pd.concat([all_gamelogs, df], ignore_index=True)
            print(f"✅ Fetched gamelogs for: {player_name}")
        except Exception as e:
            print(f"❌ Failed to fetch gamelogs for: {player_name} | Error: {e}")
            
    return all_gamelogs

def fetch_team_stats(season='2023-24'):
    """Fetches advanced team statistics for a given season."""
    try:
        print("\nFetching advanced team stats...")
        time.sleep(1)
        team_stats_raw = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense='Advanced'
        ).get_data_frames()[0]
        
        # Select and rename columns for clarity
        # Removed 'OPP_EFG_PCT' as it was causing a KeyError
        cols_to_keep = {
            'TEAM_NAME': 'TEAM_NAME',
            'TEAM_ID': 'TEAM_ID',
            'W_PCT': 'W_PCT',
            'OFF_RATING': 'OFF_RATING',
            'DEF_RATING': 'DEF_RATING',
            'PACE': 'PACE',
            'EFG_PCT': 'EFG_PCT',
            'TM_TOV_PCT': 'TM_TOV_PCT'
        }
        
        # Ensure all columns we want to keep exist in the dataframe
        available_cols = {k: v for k, v in cols_to_keep.items() if k in team_stats_raw.columns}
        team_stats = team_stats_raw[list(available_cols.keys())]
        team_stats = team_stats.rename(columns=available_cols)
        
        # Get team abbreviations
        team_info = pd.DataFrame(teams.get_teams())
        team_info = team_info[['id', 'abbreviation']]
        team_info = team_info.rename(columns={'id': 'TEAM_ID', 'abbreviation': 'TEAM_ABBREVIATION'})
        
        team_stats = pd.merge(team_stats, team_info, on='TEAM_ID')
        
        print("✅ Successfully fetched advanced team stats.")
        return team_stats
    except Exception as e:
        print(f"❌ Failed to fetch team stats | Error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)

    PLAYERS_TO_FETCH = [
        'LeBron James', 'Stephen Curry', 'Jayson Tatum', 'Kevin Durant',
        'Joel Embiid', 'Luka Doncic', 'Giannis Antetokounmpo', 'Trae Young',
        'Anthony Edwards', 'Kawhi Leonard', 'Tyrese Haliburton', 'Shai Gilgeous-Alexander'
    ]
    SEASON = '2023-24'

    # --- Fetch Player and Team Data ---
    player_gamelogs_df = fetch_player_gamelogs(PLAYERS_TO_FETCH, season=SEASON)
    team_stats_df = fetch_team_stats(season=SEASON)

    # --- Save Data ---
    if not player_gamelogs_df.empty:
        player_gamelogs_df.to_csv('data/player_game_logs.csv', index=False)
        print("\n✅ Player game logs saved to 'data/player_game_logs.csv'")
    
    if not team_stats_df.empty:
        team_stats_df.to_csv('data/team_stats.csv', index=False)
        print("✅ Team stats saved to 'data/team_stats.csv'")

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import os
import time

def get_player_id(player_name):
    player_dict = players.find_players_by_full_name(player_name)
    if player_dict:
        return player_dict[0]['id']
    else:
        raise ValueError(f"Player {player_name} not found!")

def fetch_recent_games(player_name, season='2023-24', num_games=15, sleep=1):
    player_id = get_player_id(player_name)

    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season')
    df = gamelog.get_data_frames()[0]

    # DEBUG: Print available columns for each player
    print(f"\nColumns for {player_name}:")
    print(df.columns)

    df = df.head(num_games)
    df['PLAYER_NAME'] = player_name

    # Extract opponent and home/away
    df['HOME'] = df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
    df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split()[-1])
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

    # Only keep valid columns
    expected_cols = ['PLAYER_NAME', 'GAME_DATE', 'OPPONENT', 'HOME',
                     'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS']
    keep_cols = [col for col in expected_cols if col in df.columns]
    df = df[keep_cols]

    time.sleep(sleep)
    return df

def fetch_multiple_players(player_list, season='2023-24', num_games=15):
    all_data = pd.DataFrame()

    for player in player_list:
        try:
            player_df = fetch_recent_games(player, season, num_games)
            all_data = pd.concat([all_data, player_df], ignore_index=True)
            print(f"✅ Fetched: {player}")
        except Exception as e:
            print(f"❌ Failed: {player} | Error: {e}")
            with open('data/error_log.txt', 'a') as f:
                f.write(f"{player} | {e}\n")

    return all_data

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)

    players_to_fetch = [
        'LeBron James', 'Stephen Curry', 'Jayson Tatum', 'Kevin Durant',
        'Joel Embiid', 'Luka Doncic', 'Giannis Antetokounmpo', 'Trae Young',
        'Anthony Edwards', 'Kawhi Leonard', 'Tyrese Haliburton', 'Shai Gilgeous-Alexander'
    ]

    season = '2023-24'
    num_games = 15

    df = fetch_multiple_players(players_to_fetch, season=season, num_games=num_games)

    df.to_csv('data/player_game_logs.csv', index=False)
    print("✅ Data saved to 'data/player_game_logs.csv'")

    with open('data/metadata.txt', 'w') as meta:
        meta.write(f"Fetched on: {pd.Timestamp.now()}\n")
        meta.write(f"Season: {season}\n")
        meta.write(f"Number of players: {len(players_to_fetch)}\n")
        meta.write(f"Games per player: {num_games}\n")
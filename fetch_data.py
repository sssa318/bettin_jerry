from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import time

def get_player_id(player_name):
    """Find player ID from name."""
    player_dict = players.find_players_by_full_name(player_name)
    if player_dict:
        return player_dict[0]['id']
    else:
        raise ValueError(f"Player {player_name} not found!")

def fetch_recent_games(player_name, season='2023-24', num_games=10, sleep=1):
    """Fetch recent games for a player."""
    player_id = get_player_id(player_name)
    
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season')
    df = gamelog.get_data_frames()[0]

    # Only keep recent n games
    df = df.head(num_games)

    # Add player name as a column
    df['PLAYER_NAME'] = player_name

    # Optional: Clean and keep useful columns
    df = df[['PLAYER_NAME', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS']]
    
    time.sleep(sleep)  # Avoid spamming the API
    return df

def fetch_multiple_players(player_list, season='2023-24', num_games=10):
    all_data = pd.DataFrame()

    for player in player_list:
        try:
            player_df = fetch_recent_games(player, season, num_games)
            all_data = pd.concat([all_data, player_df], ignore_index=True)
            print(f"✅ Fetched: {player}")
        except Exception as e:
            print(f"❌ Failed: {player} | Error: {e}")

    return all_data

if __name__ == "__main__":
    players_to_fetch = ['Stephen Curry', 'LeBron James', 'Jayson Tatum']  # Feel free to change
    df = fetch_multiple_players(players_to_fetch, season='2023-24', num_games=15)
    
    # Save to CSV
    df.to_csv('data/player_game_logs.csv', index=False)
    print("✅ Data saved to 'data/player_game_logs.csv'")

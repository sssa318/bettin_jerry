# 02_feature_engineering.py

import pandas as pd
import os

def load_data(player_path, team_path):
    """Loads player and team data from CSV files."""
    if not os.path.exists(player_path):
        raise FileNotFoundError(f"Could not find player dataset at {player_path}")
    if not os.path.exists(team_path):
        raise FileNotFoundError(f"Could not find team dataset at {team_path}")
        
    player_df = pd.read_csv(player_path, parse_dates=['GAME_DATE'])
    team_df = pd.read_csv(team_path)
    print("✅ Data loaded successfully.")
    return player_df, team_df

def create_features(player_df, team_df):
    """Engineers features for the model."""
    
    # Sort data chronologically for each player for rolling calculations
    player_df = player_df.sort_values(by=['PLAYER_NAME', 'GAME_DATE'])

    # --- Feature 1: Rolling Averages (Recent Form) ---
    # Using Exponentially Weighted Moving Average (EWMA) to give more weight to recent games.
    rolling_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS']
    for col in rolling_cols:
        # We use .shift(1) to ensure we're only using data from *previous* games to predict a future game
        player_df[f'{col}_ewma_3'] = player_df.groupby('PLAYER_NAME')[col].transform(
            lambda x: x.shift(1).ewm(span=3, adjust=False).mean()
        )
    print("✅ Created EWMA features for recent form.")

    # --- Feature 2: Days of Rest ---
    player_df['REST_DAYS'] = player_df.groupby('PLAYER_NAME')['GAME_DATE'].diff().dt.days
    # Fill missing rest days (e.g., for the first game of the season) with a typical value
    player_df['REST_DAYS'].fillna(3, inplace=True)
    print("✅ Created REST_DAYS feature.")

    # --- Feature 3: Opponent-Adjusted Stats ---
    # Prepare team stats for merging by renaming columns to represent the *opponent's* stats
    opponent_stats = team_df.rename(columns={
        'TEAM_ABBREVIATION': 'OPPONENT_ABBREVIATION',
        'OFF_RATING': 'OPP_OFF_RATING',
        'DEF_RATING': 'OPP_DEF_RATING',
        'PACE': 'OPP_PACE',
        'EFG_PCT': 'OPP_EFG_PCT',
        'OPP_EFG_PCT': 'OPP_OPP_EFG_PCT', # Opponent's opponent EFG%
        'TM_TOV_PCT': 'OPP_TM_TOV_PCT'
    })
    
    # Merge the opponent stats into the main dataframe
    merged_df = pd.merge(
        player_df,
        opponent_stats[['OPPONENT_ABBREVIATION', 'OPP_DEF_RATING', 'OPP_PACE', 'OPP_OFF_RATING']],
        on='OPPONENT_ABBREVIATION',
        how='left'
    )
    print("✅ Merged opponent stats to create matchup features.")
    
    return merged_df

if __name__ == "__main__":
    PLAYER_LOGS_PATH = 'data/player_game_logs.csv'
    TEAM_STATS_PATH = 'data/team_stats.csv'
    ENHANCED_DATA_PATH = 'data/enhanced_player_logs.csv'

    # Load data
    players, teams_stats = load_data(PLAYER_LOGS_PATH, TEAM_STATS_PATH)

    # Create features
    enhanced_df = create_features(players, teams_stats)

    # Save the new dataset
    enhanced_df.to_csv(ENHANCED_DATA_PATH, index=False)
    print(f"\n✅ Enhanced data with new features saved to '{ENHANCED_DATA_PATH}'")
    
    # Display a sample of the new features
    print("\n--- Sample of Enhanced Data ---")
    print(enhanced_df[['PLAYER_NAME', 'GAME_DATE', 'PTS', 'PTS_ewma_3', 'REST_DAYS', 'OPP_DEF_RATING', 'OPP_PACE']].tail())

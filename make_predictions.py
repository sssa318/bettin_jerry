# 04_make_predictions.py

import pandas as pd
import os
import joblib
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

def load_dependencies():
    """Loads the trained models and team stats data."""
    models = {}
    model_dir = 'models'
    if not os.path.exists(model_dir):
        raise FileNotFoundError("Models directory not found. Please run 03_model_training.py first.")

    for model_file in os.listdir(model_dir):
        if model_file.endswith('.joblib'):
            target_name = model_file.split('_')[0]
            models[target_name] = joblib.load(os.path.join(model_dir, model_file))
    
    team_stats_path = 'data/team_stats.csv'
    if not os.path.exists(team_stats_path):
        raise FileNotFoundError("Team stats not found. Please run 01_data_collection.py first.")
    
    team_stats = pd.read_csv(team_stats_path)
    
    print(f"✅ Loaded {len(models)} models and team stats.")
    return models, team_stats

def get_player_id(player_name):
    """Fetches a player's ID from their full name."""
    try:
        player_dict = players.find_players_by_full_name(player_name)
        return player_dict[0]['id'] if player_dict else None
    except Exception:
        return None

def prepare_prediction_data(player_name, opponent_abbr, team_stats):
    """Prepares the feature set for a single prediction."""
    print(f"\nFetching latest stats for {player_name}...")
    player_id = get_player_id(player_name)
    if not player_id:
        raise ValueError(f"Could not find player ID for {player_name}")

    # Fetch recent games to calculate EWMA
    gamelogs = playergamelog.PlayerGameLog(player_id=player_id, season='2023-24').get_data_frames()[0]
    
    # --- Create EWMA features from the player's recent history ---
    rolling_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PLUS_MINUS']
    latest_features = {}
    for col in rolling_cols:
        ewma = gamelogs[col].ewm(span=3, adjust=False).mean()
        latest_features[f'{col}_ewma_3'] = ewma.iloc[-1] # Get the most recent value

    # --- Get opponent stats ---
    opponent_data = team_stats[team_stats['TEAM_ABBREVIATION'] == opponent_abbr]
    if opponent_data.empty:
        raise ValueError(f"Could not find team stats for opponent: {opponent_abbr}")
    
    latest_features['OPP_DEF_RATING'] = opponent_data['DEF_RATING'].iloc[0]
    latest_features['OPP_PACE'] = opponent_data['PACE'].iloc[0]
    latest_features['OPP_OFF_RATING'] = opponent_data['OFF_RATING'].iloc[0]

    # --- Add other static features ---
    # Assuming the game is at a neutral site for simplicity, and average rest
    latest_features['HOME'] = 1 # Assuming a home game for this prediction
    latest_features['REST_DAYS'] = 3 # Assuming average rest

    # Create a DataFrame in the correct feature order
    feature_order = [
        'HOME', 'REST_DAYS', 'OPP_DEF_RATING', 'OPP_PACE', 'OPP_OFF_RATING',
        'PTS_ewma_3', 'REB_ewma_3', 'AST_ewma_3', 'STL_ewma_3', 
        'BLK_ewma_3', 'TOV_ewma_3', 'PLUS_MINUS_ewma_3'
    ]
    
    prediction_df = pd.DataFrame([latest_features])
    prediction_df = prediction_df[feature_order] # Ensure columns are in the same order as training
    
    print("✅ Prediction data prepared.")
    return prediction_df

def predict_player_stats(player_name, opponent_abbr, models, team_stats):
    """Predicts stats for a given player and opponent matchup."""
    try:
        # 1. Prepare the data
        prediction_input = prepare_prediction_data(player_name, opponent_abbr, team_stats)
        
        # 2. Make predictions using each model
        predictions = {}
        for stat, model in models.items():
            pred_value = model.predict(prediction_input)[0]
            predictions[stat.upper()] = pred_value
            
        return predictions

    except Exception as e:
        print(f"❌ Could not make prediction for {player_name}: {e}")
        return None

if __name__ == "__main__":
    # Load our trained models and data
    trained_models, team_statistics = load_dependencies()

    # --- Define Today's Matchups to Predict ---
    matchups = [
        {'player': 'Luka Doncic', 'opponent': 'LAC'},
        {'player': 'Jayson Tatum', 'opponent': 'MIA'},
        {'player': 'Stephen Curry', 'opponent': 'SAC'}
    ]

    print("\n--- Player Stat Predictions ---")
    for matchup in matchups:
        player = matchup['player']
        opponent = matchup['opponent']
        
        results = predict_player_stats(player, opponent, trained_models, team_statistics)
        
        if results:
            print(f"\nPrediction for {player} vs. {opponent}:")
            for stat, value in results.items():
                print(f"  - Predicted {stat}: {value:.1f}")


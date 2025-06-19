# 03_model_training.py

import pandas as pd
import xgboost as xgb
import os
import joblib # For saving the trained models

def train_models():
    """Loads enhanced data and trains a model for each target stat."""
    
    # --- 1. Load Data ---
    data_path = 'data/enhanced_player_logs.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Enhanced dataset not found at {data_path}. Please run 02_feature_engineering.py first.")
    
    df = pd.read_csv(data_path)
    print("✅ Enhanced data loaded.")
    
    # --- 2. Prepare Data for Modeling ---
    # Define which columns are features (X) and which are targets (y)
    # We include our engineered features and basic context like HOME status
    features = [
        'HOME', 'REST_DAYS', 'OPP_DEF_RATING', 'OPP_PACE', 'OPP_OFF_RATING',
        'PTS_ewma_3', 'REB_ewma_3', 'AST_ewma_3', 'STL_ewma_3', 
        'BLK_ewma_3', 'TOV_ewma_3', 'PLUS_MINUS_ewma_3'
    ]
    
    targets = ['PTS', 'REB', 'AST'] # We will train a separate model for each of these

    # Drop rows with missing values that were created by rolling averages
    df.dropna(subset=features + targets, inplace=True)
    
    X = df[features]
    y = df[targets]
    
    print(f"Data prepared. Using {len(features)} features to predict {len(targets)} targets.")
    print("Features:", features)

    # --- 3. Train a Model for Each Target ---
    os.makedirs('models', exist_ok=True) # Directory to save trained models
    
    trained_models = {}

    for target in targets:
        print(f"\n--- Training model for: {target} ---")
        
        # Define the model. XGBoost is a powerful gradient boosting model.
        # We use a regression model because we're predicting a number (e.g., points).
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000, # Number of trees to build
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=10 # Stops training if performance doesn't improve
        )
        
        # Train the model
        # Using a simple train/validation split to monitor for overfitting
        # Note: A more robust approach would use cross-validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y[target], test_size=0.2, random_state=42)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False # Set to True to see training progress
        )
        
        print(f"✅ Model for {target} trained.")
        
        # Save the trained model to a file
        model_path = f'models/{target}_predictor.joblib'
        joblib.dump(model, model_path)
        print(f"💾 Model saved to {model_path}")
        
        trained_models[target] = model

    return trained_models

if __name__ == "__main__":
    models = train_models()
    print("\n✅ All models trained and saved successfully.")


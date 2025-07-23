import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib

def train_model_no_leakage(data_path='tennis_data_features_no_leakage.csv'):
    """
    Train XGBoost model on leak-free dataset.
    
    Key differences from previous version:
    1. Uses leak-free dataset (one row per match)
    2. All target values are 1 (winner perspective)
    3. Features represent Winner vs Loser differences
    4. Proper temporal validation
    """
    print("Loading leak-free data...")
    data = pd.read_csv(data_path, low_memory=False)
    data['Date'] = pd.to_datetime(data['Date'])
    data.fillna(0, inplace=True)
    data.sort_values(by='Date', inplace=True)

    features = [
        'career_win_pct_diff',
        'surface_win_pct_diff', 
        'recent_form_diff',
        'h2h_win_pct_diff',
        'ranking_diff',
        'elo_rating_diff',
        'glicko2_rating_diff',
        'glicko2_rd_diff',
        'glicko2_volatility_diff'
    ]
    
    print("Preparing features...")
    X = data[features]
    y = data['Win']  # All values are 1, but we need to create balanced dataset
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts()}")
    
    # Create balanced dataset by flipping perspectives
    # For each match, create two rows: Winner perspective (Win=1) and Loser perspective (Win=0)
    print("Creating balanced dataset...")
    
    # Winner perspective (original data)
    X_winner = X.copy()
    y_winner = pd.Series([1] * len(X))  # Winner always wins from winner's perspective
    
    # Loser perspective (flip all features and set Win=0)
    X_loser = -X.copy()  # Flip all feature differences
    y_loser = pd.Series([0] * len(X))  # Loser always loses from loser's perspective
    
    # Combine both perspectives
    X_balanced = pd.concat([X_winner, X_loser], ignore_index=True)
    y_balanced = pd.concat([y_winner, y_loser], ignore_index=True)
    
    # Create date array for temporal validation
    dates_balanced = pd.concat([data['Date'], data['Date']], ignore_index=True)
    
    print(f"Balanced dataset shape: {X_balanced.shape}")
    print(f"Balanced target distribution: {y_balanced.value_counts()}")
    
    # Use TimeSeriesSplit for proper temporal validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize model with conservative parameters
    model = xgb.XGBClassifier(
        n_estimators=200,  # Increased from 100
        learning_rate=0.05,  # Reduced from 0.01 for faster convergence
        max_depth=4,  # Slightly increased from 3
        min_child_weight=3,  # Reduced from 5
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("\nTraining model with time series cross-validation...")
    accuracies = []
    log_losses = []
    
    # Create combined dataset with match IDs for proper splitting
    combined_data = pd.DataFrame({
        'Date': dates_balanced,
        'match_id': list(range(len(X))) + list(range(len(X)))  # Same match_id for both perspectives
    })
    combined_data = pd.concat([combined_data, X_balanced], axis=1)
    combined_data['target'] = y_balanced
    
    # Get unique matches for time series split
    unique_matches = data[['Date']].reset_index().rename(columns={'index': 'match_id'})
    unique_matches.sort_values('Date', inplace=True)
    
    # Perform time series cross-validation on matches (not individual rows)
    tscv_matches = TimeSeriesSplit(n_splits=5)
    
    for fold, (train_match_idx, test_match_idx) in enumerate(tscv_matches.split(unique_matches), 1):
        # Get match IDs for train and test
        train_match_ids = unique_matches.iloc[train_match_idx]['match_id'].tolist()
        test_match_ids = unique_matches.iloc[test_match_idx]['match_id'].tolist()
        
        # Filter balanced dataset based on match IDs
        train_mask = combined_data['match_id'].isin(train_match_ids)
        test_mask = combined_data['match_id'].isin(test_match_ids)
        
        X_train = combined_data[train_mask][features]
        y_train = combined_data[train_mask]['target']
        X_test = combined_data[test_mask][features]
        y_test = combined_data[test_mask]['target']
        
        # Get date range for this fold
        train_dates = combined_data[train_mask]['Date']
        test_dates = combined_data[test_mask]['Date']
        print(f"\nFold {fold}")
        print(f"Training period: {train_dates.min()} to {train_dates.max()}")
        print(f"Testing period: {test_dates.min()} to {test_dates.max()}")
        print(f"Training samples: {len(X_train)} (matches: {len(train_match_ids)})")
        print(f"Test samples: {len(X_test)} (matches: {len(test_match_ids)})")
        print(f"Train target distribution: {y_train.value_counts().to_dict()}")
        print(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        # Train the model
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False  # Reduced verbosity
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        
        accuracies.append(accuracy)
        log_losses.append(logloss)
        
        print(f"Fold {fold} - Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}")
    
    print(f"\nCross-Validation Results:")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
    print(f"Average Log Loss: {np.mean(log_losses):.4f} (±{np.std(log_losses):.4f})")
    
    # Final fit on all balanced data
    print("\nTraining final model on all balanced data...")
    model.fit(X_balanced, y_balanced)
    
    # Feature importance analysis
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (No Data Leakage):")
    print("=" * 50)
    for _, row in importance.iterrows():
        print(f"{row['feature']:<30} {row['importance']:.4f}")
    
    # Additional analysis: test on original unbalanced data to see real-world performance
    print("\nTesting on original unbalanced data (winner perspective only)...")
    y_pred_original = model.predict(X)
    y_pred_proba_original = model.predict_proba(X)[:, 1]
    
    print(f"Prediction distribution: {pd.Series(y_pred_original).value_counts()}")
    print(f"Probability stats: mean={y_pred_proba_original.mean():.4f}, "
          f"std={y_pred_proba_original.std():.4f}")
    print(f"Probability range: {y_pred_proba_original.min():.4f} to {y_pred_proba_original.max():.4f}")
    
    print("\nSaving model...")
    # Save the model trained on balanced data
    joblib.dump(model, 'tennis_model_no_leakage.joblib')
    
    # Save feature importance for future reference
    importance.to_csv('feature_importance_no_leakage.csv', index=False)
    
    print("Leak-free model and feature importance saved successfully!")
    
    return model, importance

if __name__ == '__main__':
    model, importance = train_model_no_leakage()

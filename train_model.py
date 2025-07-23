import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import joblib

def train_model():
    """Train tennis prediction model."""
    print("Loading data...")
    
    df = pd.read_csv('tennis_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    features = [
        'career_win_pct_diff', 'surface_win_pct_diff', 'recent_form_diff',
        'h2h_win_pct_diff', 'elo_rating_diff',
        'glicko2_rating_diff', 'glicko2_rd_diff', 'glicko2_volatility_diff'
    ]
    
    X = df[features]
    y = df['Win']
    
    print(f"Dataset: {len(df)} matches, {len(features)} features")
    
    # Temporal split (80% train, 20% test)
    split_date = df['Date'].quantile(0.8)
    train_mask = df['Date'] <= split_date
    
    # Balance training data (winner + loser perspectives)
    X_train_orig = X[train_mask]
    y_train_orig = y[train_mask]
    
    # Create balanced dataset - flip features for losers  
    X_train = pd.concat([X_train_orig, -X_train_orig])
    y_train = pd.concat([y_train_orig, pd.Series([0]*len(y_train_orig), index=y_train_orig.index)])
    
    # Test on original data (all wins)
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # For test set with only class 1, calculate metrics differently
    if len(set(y_test)) == 1:
        # Only accuracy makes sense when test has one class
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Test set contains only winners (class 1)")
        print(f"Average prediction probability: {y_prob.mean():.4f}")
    else:
        logloss = log_loss(y_test, y_prob)
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Log Loss: {logloss:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"{row['feature']:<25} {row['importance']:.4f}")
    
    # Save
    joblib.dump(model, 'tennis_model.joblib')
    importance.to_csv('feature_importance.csv', index=False)
    
    print("\nModel saved!")
    return model

if __name__ == "__main__":
    train_model()

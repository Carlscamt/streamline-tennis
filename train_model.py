
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import joblib
from features import FEATURES

def train_model():
    """Train tennis prediction model."""
    print("Loading data...")
    
    df = pd.read_csv('tennis_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use centralized feature list
    X = df[FEATURES]
    y = df['Win']
    
    print(f"Dataset: {len(df)} matches, {len(FEATURES)} features")
    
    # Proper match-level split: ensure both perspectives of a match are in the same set
    if 'match_id' in df.columns:
        unique_matches = df['match_id'].unique()
        unique_matches.sort()
        split_idx = int(len(unique_matches) * 0.8)
        train_matches = unique_matches[:split_idx]
        test_matches = unique_matches[split_idx:]
        train_mask = df['match_id'].isin(train_matches)
        test_mask = df['match_id'].isin(test_matches)
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
    else:
        # Fallback: temporal split (should not be used with dyadic data)
        split_date = df['Date'].quantile(0.8)
        train_mask = df['Date'] <= split_date
        X_train = X[train_mask]
        y_train = y[train_mask]
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
        'feature': FEATURES,
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

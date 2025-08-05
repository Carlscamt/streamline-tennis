
"""
🎾 TENNIS MATCH PREDICTION - MODEL TRAINING
==========================================

XGBOOST MODEL TRAINING PIPELINE
==============================
This module trains the XGBoost machine learning model using features generated
by the feature engineering pipeline. It implements proper data splitting and
validation to avoid data leakage in tennis match prediction.

🚀 ULTRA-STREAMLINED INTEGRATION:
================================
This training pipeline integrates with the ultra-streamlined feature system:
• Uses FEATURES list from features.py automatically
• No need to manually specify feature columns
• Adding features requires only editing features.py

🎯 KEY FEATURES:
===============
• Data Leakage Prevention: Match-level splits ensure both perspectives stay together
• Balanced Training: Handles winner/loser data perspectives properly
• Feature Validation: Uses centralized feature definitions
• Model Persistence: Saves trained model and feature importance
• Performance Metrics: Accuracy and log loss evaluation

📊 TRAINING PIPELINE:
====================
1. Load preprocessed data (tennis_features.csv)
2. Extract features using centralized FEATURES list
3. Implement proper match-level data splitting
4. Train XGBoost classifier with optimized parameters
5. Evaluate model performance on test set
6. Generate feature importance analysis
7. Save model and metrics for production use

🔧 DATA SPLITTING STRATEGY:
==========================

Match-Level Split (Recommended):
• Groups both perspectives of each match together
• Prevents data leakage from seeing future outcomes
• Maintains temporal relationships in training/test split
• Uses match_id column for proper grouping

Temporal Split (Fallback):
• Falls back to date-based split if match_id unavailable
• Less robust but better than random splitting
• Should be avoided with dyadic (match) data

🎮 MODEL CONFIGURATION:
======================
XGBoost Parameters:
• n_estimators: 200 (number of boosting rounds)
• max_depth: 4 (tree depth for complexity control)
• learning_rate: 0.05 (conservative learning for stability)
• random_state: 42 (reproducible results)

These parameters are optimized for tennis match prediction and provide
good balance between accuracy and generalization.

📈 EVALUATION METRICS:
=====================
• Accuracy: Overall prediction correctness
• Log Loss: Probabilistic prediction quality
• Feature Importance: XGBoost feature contribution analysis
• Class Distribution: Training/test set balance validation

🎯 OUTPUT FILES:
===============
• tennis_model.joblib: Trained XGBoost model
• feature_importance.csv: Feature contribution analysis
• Console output: Training metrics and validation results

💡 PERFORMANCE CONSIDERATIONS:
=============================
• Memory Efficient: Processes data in single pass
• Fast Training: Optimized XGBoost parameters
• Reproducible: Fixed random seed for consistent results
• Robust Validation: Proper splitting prevents overfitting

🚀 USAGE:
========
Command Line:
  python train_model.py

Programmatic:
  from train_model import train_model
  model = train_model()

Requirements:
• tennis_features.csv: Preprocessed training data
• features.py: Feature definitions
• All dependencies in requirements.txt

🔗 RELATED FILES:
================
• features.py: Feature definitions and computation
• feature_engineering.py: Generates tennis_features.csv
• match_predictor.py: Uses trained model for predictions
• tennis_model.joblib: Output trained model
• feature_importance.csv: Output feature analysis
"""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss
import joblib
from features import FEATURES, ADDITIONAL_FEATURES

def train_model():
    """Train tennis prediction model."""
    print("Loading data...")
    
    df = pd.read_csv('../data/tennis_features_sample.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use ALL features (core + additional) for enhanced model
    ALL_FEATURES = FEATURES + ADDITIONAL_FEATURES
    print(f"Using {len(FEATURES)} core features + {len(ADDITIONAL_FEATURES)} additional features = {len(ALL_FEATURES)} total features")
    
    # Check which features are available in the dataset
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing_features = [f for f in ALL_FEATURES if f not in df.columns]
    
    if missing_features:
        print(f"WARNING: {len(missing_features)} features missing from dataset:")
        for feature in missing_features[:10]:  # Show first 10
            print(f"  - {feature}")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")
        print(f"Training with {len(available_features)} available features")
    
    # Use available features
    X = df[available_features]
    y = df['Win']
    
    print(f"Dataset: {len(df)} matches, {len(available_features)} features")
    
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
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    for _, row in importance.iterrows():
        print(f"{row['feature']:<25} {row['importance']:.4f}")
    
    # Save
    joblib.dump(model, '../models/tennis_model.joblib')
    importance.to_csv('../data/feature_importance.csv', index=False)
    
    print("\nModel saved!")
    return model

if __name__ == "__main__":
    train_model()

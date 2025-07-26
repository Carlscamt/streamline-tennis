

"""
🎾 TENNIS MATCH PREDICTION - BACKTESTING & VALIDATION
====================================================

WALK-FORWARD VALIDATION FOR MODEL PERFORMANCE
============================================
This module implements comprehensive backtesting of the tennis prediction model
using walk-forward validation. It tests model performance across multiple years
and analyzes feature drift to ensure robust predictions.

🎯 BACKTESTING METHODOLOGY:
==========================
• Walk-Forward Validation: Train on historical data, test on future years
• Temporal Splits: Maintains chronological data integrity
• Feature Drift Analysis: Detects changes in feature distributions
• Multi-Year Testing: Validates consistency across 2023, 2024, 2025

📊 VALIDATION STRATEGY:
======================

Time-Based Splitting:
• Training: All data before test year
• Testing: Single year of data
• Progressive: Each year uses more training data
• Realistic: Simulates real-world deployment

Feature Drift Detection:
• Compares feature means between train/test sets
• Flags significant distribution changes (>15% difference)
• Helps identify when model retraining is needed
• Ensures feature stability over time

🎮 PERFORMANCE METRICS:
======================
• Accuracy: Overall prediction correctness
• Log Loss: Probabilistic prediction quality
• ROC AUC: Ranking ability across probability spectrum
• Classification Report: Precision, recall, F1-score breakdown
• Feature Importance: Contribution analysis by year

🔧 MODEL CONFIGURATION:
======================
Enhanced XGBoost Parameters:
• n_estimators: 800 (more trees for complex patterns)
• max_depth: 4 (controlled complexity)
• learning_rate: 0.05 (conservative learning)
• subsample: 0.8 (prevents overfitting)
• colsample_bytree: 0.8 (feature randomization)
• reg_alpha/lambda: L1/L2 regularization
• Evaluation metrics: logloss, error, auc

🚀 ULTRA-STREAMLINED INTEGRATION:
================================
Uses the ultra-streamlined feature system:
• Automatically uses FEATURES from features.py
• No manual feature specification required
• Consistent with production model training

📈 OUTPUT ANALYSIS:
==================
For Each Test Year:
• Overall accuracy and loss metrics
• Detailed classification performance
• Feature importance rankings
• Feature drift warnings
• Model evaluation details

Summary Results:
• Year-over-year performance trends
• Feature stability analysis
• Model robustness assessment

💡 INSIGHTS PROVIDED:
====================
• Model Consistency: Performance across different time periods
• Feature Stability: Which features remain predictive over time
• Drift Detection: When features change distribution
• Seasonality Effects: Year-specific patterns in tennis

🎯 USAGE:
========
Command Line:
  python backtest_2025.py

Output:
• Console: Detailed metrics and analysis
• Validation of model robustness over time
• Feature drift warnings for maintenance

🔧 CUSTOMIZATION:
================
• Test Years: Modify test_years list for different periods
• Metrics: Add/remove evaluation metrics
• Thresholds: Adjust drift detection sensitivity
• Model Parameters: Tune XGBoost configuration

🔗 RELATED FILES:
================
• features.py: Feature definitions used in backtesting
• train_model.py: Production model training
• tennis_features.csv: Source data for validation
• feature_importance.csv: Compare with backtest importance
"""

import pandas as pd
import numpy as np
from .features import FEATURES
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
import xgboost as xgb

# Load features with sorting
df = pd.read_csv('../data/tennis_features.csv', parse_dates=['Date'])
df = df.sort_values('Date')  # Critical for temporal splits

def walk_forward_validation():
    test_years = [2023, 2024, 2025]
    results = []
    for year in test_years:
        train = df[df['Date'] < f'{year}-01-01']
        test = df[df['Date'].dt.year == year]
        if len(test) == 0 or len(train) == 0:
            continue

        X_train, y_train = train[FEATURES], train['Win']
        X_test, y_test = test[FEATURES], test['Win']

        model = xgb.XGBClassifier(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=42,
            use_label_encoder=False,
            eval_metric=['logloss', 'error', 'auc']
        )
        model.fit(X_train, y_train, verbose=10)

        # Feature drift check
        print(f"\nFeature Drift Analysis for {year}:")
        for feature in FEATURES:
            t_mean, v_mean = X_train[feature].mean(), X_test[feature].mean()
            drift = abs(t_mean - v_mean) / max(1e-9, abs(t_mean))
            if drift > 0.15:
                print(f"⚠️ {feature}: Train={t_mean:.4f}, Test={v_mean:.4f} ({drift:.1%} diff)")

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f'\n{year} Backtest Results:')
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('Log Loss:', log_loss(y_test, y_proba))
        print('ROC AUC:', roc_auc_score(y_test, y_proba))
        print(classification_report(y_test, y_pred))

        importances = model.feature_importances_
        for name, val in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
            print(f'{name}: {val:.4f}')
        results.append((year, accuracy_score(y_test, y_pred)))
    return results

if __name__ == "__main__":
    walk_forward_validation()
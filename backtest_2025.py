

import pandas as pd
import numpy as np
from features import FEATURES
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, classification_report
import xgboost as xgb

# Load features with sorting
df = pd.read_csv('tennis_features.csv', parse_dates=['Date'])
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
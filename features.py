# Centralized feature list for tennis match prediction pipeline
#
# To add a new feature:
#   1. Add the feature name to the FEATURES list below.
#   2. Implement the logic for computing this feature in feature_engineering.py (and any other relevant scripts).
#   3. The training, prediction, and feature engineering scripts will automatically use the updated list.
#
# This ensures a single source of truth for all model features.

FEATURES = [
    'career_win_pct_diff',
    'surface_win_pct_diff',
    'recent_form_diff',
    'h2h_win_pct_diff',
    'elo_rating_diff',
    'glicko2_rating_diff',
    'glicko2_rd_diff',
    'glicko2_volatility_diff',
]

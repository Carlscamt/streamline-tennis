"""
🎾 TENNIS MATCH PREDICTION - FEATURE SYSTEM
=========================================

This is the SINGLE SOURCE OF TRUTH for all tennis match prediction features.
The entire pipeline (training, prediction, and web app) uses this file.

🚀 ULTRA-STREAMLINED WORKFLOW FOR ADDING FEATURES:
================================================
To add a new feature, edit ONLY this file:
  1. Add the feature name to the FEATURES list (for current model) or ADDITIONAL_FEATURES
  2. Add the computation logic to the compute_feature() function
  3. That's it! The feature flows through the entire pipeline automatically.

📋 FEATURE CATEGORIES:
====================
• Performance Metrics: Win percentages, recent form
• Rating Systems: ELO and Glicko-2 ratings
• Head-to-Head: Historical matchup data
• Surface Analysis: Court surface performance
• Temporal Factors: Fatigue, streaks, adaptability

🔧 ARCHITECTURE:
===============
• FEATURES: Core 8 features used by the trained model
• ADDITIONAL_FEATURES: Extended features (require model retraining)
• compute_feature(): Single function computing all features
• Helper functions: Build feature vectors for training and prediction

📊 FEATURE DESCRIPTIONS:
=======================
Core Features (8 - used by current model):
• career_win_pct_diff: Difference in overall career win percentages
• surface_win_pct_diff: Difference in win percentages on current surface
• recent_form_diff: Difference in recent match performance (last 10 matches)
• h2h_win_pct_diff: Difference in head-to-head win percentage
• elo_rating_diff: Difference in ELO ratings
• glicko2_rating_diff: Difference in Glicko-2 ratings
• glicko2_rd_diff: Difference in Glicko-2 rating deviations (uncertainty)
• glicko2_volatility_diff: Difference in Glicko-2 volatility scores

Additional Features (5 - require retraining):
• fatigue_days_diff: Difference in days since last match
• h2h_surface_win_pct_diff: Head-to-head win rate on current surface
• surface_adaptability_diff: Adaptability to different court surfaces
• win_streak_diff: Difference in current winning streaks
• loss_streak_diff: Difference in current losing streaks

🎯 USAGE EXAMPLES:
=================
Training Data Generation:
  context = {'winner_data': {...}, 'loser_data': {...}}
  features = build_feature_vector(winner_data, loser_data)

Real-time Prediction:
  context = {'player1_data': {...}, 'player2_data': {...}}
  features = build_prediction_feature_vector(p1_data, p2_data)

Single Feature Computation:
  value = compute_feature('elo_rating_diff', context)

💡 EXTENDING THE SYSTEM:
=======================
1. Add new feature name to ADDITIONAL_FEATURES list
2. Implement computation logic in compute_feature() function
3. Test with existing prediction pipeline
4. Retrain model to use new features in production

🔗 RELATED FILES:
================
• feature_builder.py: Lightweight wrapper for backward compatibility
• feature_engineering.py: Uses this for training data generation
• match_predictor.py: Uses this for real-time predictions
• train_model.py: Trains model using features defined here
"""

import numpy as np
from typing import Dict, Any

# ============================================================================
# FEATURE REGISTRY - Add new features here
# ============================================================================

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

# Additional features that can be computed but aren't used by the current model
# To use these, you would need to retrain the model with the expanded feature set
ADDITIONAL_FEATURES = [
    'fatigue_days_diff',
    'h2h_surface_win_pct_diff',
    'surface_adaptability_diff',
    'win_streak_diff',
    'loss_streak_diff',
]

# ============================================================================
# FEATURE COMPUTATION LOGIC - Add new feature calculations here
# ============================================================================

def compute_feature(feature_name: str, context: Dict[str, Any]) -> float:
    """
    Compute a single feature value given the feature name and context.
    
    Args:
        feature_name: Name of the feature to compute (must be in FEATURES or ADDITIONAL_FEATURES)
        context: Dictionary containing all necessary data for feature computation
        
    Returns:
        float: The computed feature value
        
    Raises:
        ValueError: If feature_name is not in FEATURES or ADDITIONAL_FEATURES list or not implemented
        KeyError: If required context data is missing
    """
    all_available_features = FEATURES + ADDITIONAL_FEATURES
    if feature_name not in all_available_features:
        raise ValueError(f"Feature '{feature_name}' not found in FEATURES or ADDITIONAL_FEATURES lists")
    
    # Extract player data from context
    # For training data generation (feature_engineering.py)
    winner_data = context.get('winner_data', {})
    loser_data = context.get('loser_data', {})
    
    # For real-time predictions (match_predictor.py)  
    player1_data = context.get('player1_data', {})
    player2_data = context.get('player2_data', {})
    
    # Use winner/loser if available, otherwise use player1/player2
    p1_data = winner_data if winner_data else player1_data
    p2_data = loser_data if loser_data else player2_data
    
    # ========================================================================
    # FEATURE IMPLEMENTATIONS - Add new features here
    # ========================================================================
    
    if feature_name == 'career_win_pct_diff':
        return p1_data['career_win_pct'] - p2_data['career_win_pct']
    
    elif feature_name == 'surface_win_pct_diff':
        return p1_data['surface_win_pct'] - p2_data['surface_win_pct']
    
    elif feature_name == 'recent_form_diff':
        return p1_data['recent_form'] - p2_data['recent_form']
    
    elif feature_name == 'h2h_win_pct_diff':
        return p1_data['h2h_win_pct'] - p2_data['h2h_win_pct']
    
    elif feature_name == 'elo_rating_diff':
        return p1_data['elo_rating'] - p2_data['elo_rating']
    
    elif feature_name == 'glicko2_rating_diff':
        return p1_data['glicko2_rating'] - p2_data['glicko2_rating']
    
    elif feature_name == 'glicko2_rd_diff':
        # Lower RD is better, so we flip the sign
        return p2_data['glicko2_rd'] - p1_data['glicko2_rd']
    
    elif feature_name == 'glicko2_volatility_diff':
        # Lower volatility is better, so we flip the sign  
        return p2_data['glicko2_volatility'] - p1_data['glicko2_volatility']
    
    elif feature_name == 'fatigue_days_diff':
        return p1_data['fatigue_days'] - p2_data['fatigue_days']
    
    elif feature_name == 'h2h_surface_win_pct_diff':
        return p1_data['h2h_surface_win_pct'] - p2_data['h2h_surface_win_pct']
    
    elif feature_name == 'surface_adaptability_diff':
        return p1_data['surface_adaptability'] - p2_data['surface_adaptability']
    
    elif feature_name == 'win_streak_diff':
        return p1_data['win_streak'] - p2_data['win_streak']
    
    elif feature_name == 'loss_streak_diff':
        return p1_data['loss_streak'] - p2_data['loss_streak']
    
    # ========================================================================
    # NEW FEATURES GO HERE - Example:
    # ========================================================================
    # elif feature_name == 'ranking_diff':
    #     return p1_data['ranking'] - p2_data['ranking']
    #
    # elif feature_name == 'age_diff':
    #     return p1_data['age'] - p2_data['age']
    # ========================================================================
    
    else:
        raise ValueError(f"Feature computation not implemented for '{feature_name}'. "
                        f"Please add implementation in features.py compute_feature() function.")


# ============================================================================
# HELPER FUNCTIONS - Used by feature_builder.py
# ============================================================================

def build_feature_vector(winner_context: Dict[str, Any], loser_context: Dict[str, Any], 
                        win_label: int = 1, extra_fields: Dict[str, Any] = None,
                        feature_list: list = None) -> Dict[str, Any]:
    """
    Build a complete feature vector for a match.
    
    Args:
        winner_context: Context data for the winning player
        loser_context: Context data for the losing player
        win_label: Label for the match outcome (1 for winner, 0 for loser)
        extra_fields: Additional fields to include (Date, match_id, etc.)
        feature_list: List of features to include (defaults to FEATURES)
        
    Returns:
        Dict containing all feature values plus any extra fields
    """
    if feature_list is None:
        feature_list = FEATURES
        
    context = {
        'winner_data': winner_context,
        'loser_data': loser_context
    }
    
    feature_vector = {}
    
    # Compute all features dynamically
    for feature_name in feature_list:
        feature_vector[feature_name] = compute_feature(feature_name, context)
        # Handle NaN values
        feature_vector[feature_name] = np.nan_to_num(feature_vector[feature_name], nan=0.0)
    
    # Add the win label
    feature_vector['Win'] = win_label
    
    # Add any extra fields
    if extra_fields:
        feature_vector.update(extra_fields)
    
    return feature_vector


def build_prediction_feature_vector(player1_context: Dict[str, Any], 
                                   player2_context: Dict[str, Any],
                                   feature_list: list = None) -> Dict[str, float]:
    """
    Build a feature vector for real-time prediction (UI).
    
    Args:
        player1_context: Context data for player 1
        player2_context: Context data for player 2
        feature_list: List of features to include (defaults to FEATURES)
        
    Returns:
        Dict containing all feature values for prediction
    """
    if feature_list is None:
        feature_list = FEATURES
        
    context = {
        'player1_data': player1_context,
        'player2_data': player2_context
    }
    
    feature_vector = {}
    
    # Compute all features dynamically
    for feature_name in feature_list:
        feature_vector[feature_name] = compute_feature(feature_name, context)
        # Handle NaN values
        feature_vector[feature_name] = np.nan_to_num(feature_vector[feature_name], nan=0.0)
    
    return feature_vector

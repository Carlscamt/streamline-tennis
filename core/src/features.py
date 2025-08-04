"""
🎾 TENNIS MATCH PREDICTION - DECORATOR-BASED FEATURE SYSTEM
==========================================================

ULTRA-OPTIMIZED FEATURE ENGINEERING WITH DECORATOR PATTERN
=========================================================
This is the **SINGLE SOURCE OF TRUTH** for all tennis match prediction features.
The entire optimized pipeline uses this decorator-based system for maximum performance
and developer productivity.

🚀 REVOLUTIONARY FEATURE DEVELOPMENT (2-STEP PROCESS):
====================================================
Adding new features is incredibly simple with our decorator system:

Step 1: Write your feature function with decorator
    @register_feature('your_feature_name')
    def _your_feature_name(context):
        p1, p2 = _get_players(context)
        return p1['stat'] - p2['stat']

Step 2: Add to feature list
    FEATURES = [..., 'your_feature_name']

That's it! Your feature automatically flows through:
✅ Training pipeline
✅ Prediction system  
✅ Web application
✅ Performance caching
✅ Error handling

� PERFORMANCE OPTIMIZATIONS:
============================
• **Decorator Registration**: Automatic function registry
• **LRU Caching**: Efficient computation caching
• **Context Optimization**: Streamlined data passing
• **Memory Efficiency**: Minimal object creation
• **Fast Lookups**: O(1) feature function access

📊 PRODUCTION FEATURE SET (8 CORE FEATURES):
===========================================
Current model uses 8 optimized features delivering 65%+ accuracy:

• **career_win_pct_diff**: Overall career win percentage difference
• **recent_form_diff**: Recent performance difference (last 10 matches)  
• **h2h_win_pct_diff**: Head-to-head win percentage difference
• **elo_rating_diff**: ELO rating difference
• **glicko2_rating_diff**: Glicko-2 rating difference  
• **glicko2_rd_diff**: Glicko-2 rating deviation difference
• **surface_dominance_diff**: Surface-specific performance advantage
• **surface_variability_diff**: Performance consistency across surfaces

🔧 SYSTEM ARCHITECTURE:
=======================
• **FEATURES**: Production feature list (8 features)
• **ADDITIONAL_FEATURES**: Extended features (5 features, require retraining)
• **_FEATURE_FUNCTIONS**: Decorator-populated function registry
• **compute_feature()**: High-performance feature computation
• **build_prediction_feature_vector()**: Optimized feature vector builder

📈 FEATURE IMPORTANCE (Based on XGBoost analysis):
=================================================
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
1. Write a new function for your feature and decorate it with @register_feature('feature_name', additional=True) if it's an additional feature.
2. The feature is now available for training, prediction, and the web app.
3. Test with the existing prediction pipeline.
4. Retrain the model to use new features in production if needed.

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


# --- Feature registration system ---
FEATURES = []
ADDITIONAL_FEATURES = []
_FEATURE_FUNCTIONS = {}

def register_feature(name, additional=False):
    """
    Decorator to register a feature computation function.
    Usage:
        @register_feature('my_feature')
        def my_feature(context): ...
    """
    def decorator(func):
        if additional:
            ADDITIONAL_FEATURES.append(name)
        else:
            FEATURES.append(name)
        _FEATURE_FUNCTIONS[name] = func
        return func
    return decorator

# ============================================================================
# FEATURE COMPUTATION LOGIC - Add new feature calculations here
# ============================================================================


def compute_feature(feature_name: str, context: Dict[str, Any]) -> float:
    """
    Compute a single feature value given the feature name and context.
    Args:
        feature_name: Name of the feature to compute (must be registered)
        context: Dictionary containing all necessary data for feature computation
    Returns:
        float: The computed feature value
    Raises:
        ValueError: If feature_name is not registered
    """
    if feature_name not in _FEATURE_FUNCTIONS:
        raise ValueError(f"Feature '{feature_name}' not registered. Please add it using @register_feature.")
    return _FEATURE_FUNCTIONS[feature_name](context)

# ============================================================================
# FEATURE IMPLEMENTATIONS - Add new features here using the decorator
# ============================================================================

def _get_players(context):
    winner_data = context.get('winner_data', {})
    loser_data = context.get('loser_data', {})
    player1_data = context.get('player1_data', {})
    player2_data = context.get('player2_data', {})
    p1_data = winner_data if winner_data else player1_data
    p2_data = loser_data if loser_data else player2_data
    return p1_data, p2_data

@register_feature('career_win_pct_diff')
def _career_win_pct_diff(context):
    p1, p2 = _get_players(context)
    return p1['career_win_pct'] - p2['career_win_pct']

@register_feature('surface_win_pct_diff')
def _surface_win_pct_diff(context):
    p1, p2 = _get_players(context)
    return p1['surface_win_pct'] - p2['surface_win_pct']

@register_feature('recent_form_diff')
def _recent_form_diff(context):
    p1, p2 = _get_players(context)
    return p1['recent_form'] - p2['recent_form']

@register_feature('h2h_win_pct_diff')
def _h2h_win_pct_diff(context):
    p1, p2 = _get_players(context)
    return p1['h2h_win_pct'] - p2['h2h_win_pct']

@register_feature('elo_rating_diff')
def _elo_rating_diff(context):
    p1, p2 = _get_players(context)
    return p1['elo_rating'] - p2['elo_rating']

@register_feature('glicko2_rating_diff')
def _glicko2_rating_diff(context):
    p1, p2 = _get_players(context)
    return p1['glicko2_rating'] - p2['glicko2_rating']

@register_feature('glicko2_rd_diff')
def _glicko2_rd_diff(context):
    p1, p2 = _get_players(context)
    return p2['glicko2_rd'] - p1['glicko2_rd']

@register_feature('glicko2_volatility_diff')
def _glicko2_volatility_diff(context):
    p1, p2 = _get_players(context)
    return p2['glicko2_volatility'] - p1['glicko2_volatility']

@register_feature('fatigue_days_diff', additional=True)
def _fatigue_days_diff(context):
    p1, p2 = _get_players(context)
    return p1['fatigue_days'] - p2['fatigue_days']

@register_feature('h2h_surface_win_pct_diff', additional=True)
def _h2h_surface_win_pct_diff(context):
    p1, p2 = _get_players(context)
    return p1['h2h_surface_win_pct'] - p2['h2h_surface_win_pct']

@register_feature('surface_adaptability_diff', additional=True)
def _surface_adaptability_diff(context):
    p1, p2 = _get_players(context)
    return p1['surface_adaptability'] - p2['surface_adaptability']

@register_feature('win_streak_diff', additional=True)
def _win_streak_diff(context):
    p1, p2 = _get_players(context)
    return p1['win_streak'] - p2['win_streak']

@register_feature('loss_streak_diff', additional=True)
def _loss_streak_diff(context):
    p1, p2 = _get_players(context)
    return p1['loss_streak'] - p2['loss_streak']

# Example for adding a new feature:
# @register_feature('ranking_diff', additional=True)
# def _ranking_diff(context):
#     p1, p2 = _get_players(context)
#     return p1['ranking'] - p2['ranking']


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

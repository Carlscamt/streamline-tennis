"""
🎾 Tennis Match Prediction System - Core Package
===============================================

This package contains the core components of the tennis match prediction system:

Modules:
    features: Ultra-streamlined feature system (central registry + computation)
    feature_builder: Backward compatibility wrapper
    feature_engineering: ELO & Glicko-2 rating systems
    train_model: XGBoost model training pipeline
    betting_strategy: Kelly Criterion betting recommendations
    backtest_2025: Walk-forward validation system
"""

__version__ = "1.0.0"
__author__ = "Tennis Prediction System"
__description__ = "Professional tennis match prediction using ML and advanced rating systems"

# Core components
from .features import FEATURES, compute_feature, build_feature_vector, build_prediction_feature_vector
from .feature_builder import *
from .betting_strategy import UncertaintyShrinkageBetting

__all__ = [
    'FEATURES',
    'compute_feature', 
    'build_feature_vector',
    'build_prediction_feature_vector',
    'UncertaintyShrinkageBetting'
]

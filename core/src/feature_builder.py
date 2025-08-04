"""
🎾 TENNIS MATCH PREDICTION - FEATURE BUILDER
==========================================

BACKWARD COMPATIBILITY WRAPPER
=============================
This module provides backward compatibility for the tennis match prediction system.
All actual feature computation has been moved to features.py as part of the 
ultra-streamlined workflow implementation.

🚀 ULTRA-STREAMLINED WORKFLOW:
============================
• All feature logic is now centralized in features.py
• Adding features requires editing ONLY features.py
• This file simply re-exports functions for compatibility

📁 ARCHITECTURE:
===============
BEFORE (Complex):
  ├── feature_builder.py (contained computation logic)
  ├── features.py (just feature names)
  └── Multiple files to edit for new features

AFTER (Ultra-Streamlined):
  ├── features.py (feature names + computation logic)
  ├── feature_builder.py (compatibility wrapper) ← YOU ARE HERE
  └── Single file to edit for new features

🔧 USAGE:
========
This module is automatically imported by:
• feature_engineering.py (training pipeline)
• Older scripts that expect feature_builder imports

You can use functions directly from features.py or through this wrapper:
  from feature_builder import FEATURES, compute_feature
  # OR
  from features import FEATURES, compute_feature

🎯 EXPORTS:
==========
• FEATURES: List of core features used by trained model
• compute_feature(): Compute individual feature values
• build_feature_vector(): Build complete feature vectors for training
• build_prediction_feature_vector(): Build feature vectors for prediction

💡 FOR DEVELOPERS:
=================
• To add features: Edit features.py (not this file)
• This file exists only for compatibility
• All computation logic is in features.py
• New projects should import directly from features.py

🔗 RELATED FILES:
================
• features.py: Contains all actual computation logic
• feature_engineering.py: Uses this for training data
• match_predictor.py: Uses features.py directly
"""

from features import FEATURES, compute_feature, build_feature_vector, build_prediction_feature_vector
import numpy as np
from typing import Dict, Any

# Re-export the functions from features.py for backward compatibility
__all__ = ['compute_feature', 'build_feature_vector', 'build_prediction_feature_vector', 'FEATURES']

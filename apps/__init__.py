"""
🎾 Tennis Match Prediction - Applications Package
===============================================

This package contains the production-ready tennis match prediction applications.

Available Applications:
- optimized_match_predictor.py: High-performance app (300x faster)
- match_predictor.py: Original app (legacy support)
"""

__version__ = "2.0.0"
__author__ = "Tennis Prediction Team"

# Make main functions available at package level
from .optimized_match_predictor import main as run_optimized_app

__all__ = ['run_optimized_app']

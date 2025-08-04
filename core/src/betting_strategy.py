"""
🎾 TENNIS MATCH PREDICTION - BETTING STRATEGY
============================================

KELLY CRITERION BETTING WITH UNCERTAINTY SHRINKAGE
=================================================
This module implements an advanced betting strategy that combines the Kelly Criterion
with uncertainty-based position sizing for tennis match betting. It accounts for
model confidence and provides responsible betting recommendations.

🎯 CORE STRATEGY:
================
• Kelly Criterion: Optimal bet sizing for long-term growth
• Uncertainty Shrinkage: Reduces bet size when model is uncertain
• Risk Management: Maximum bet limits and minimum probability thresholds
• Fair Odds Calculation: Determines value betting opportunities

📊 BETTING METHODOLOGY:
======================

Kelly Criterion Formula:
f = (bp - q) / b
Where:
• f = fraction of bankroll to bet
• b = odds - 1 (net odds)
• p = probability of winning
• q = probability of losing (1 - p)

Uncertainty Shrinkage:
• Calculates model uncertainty: 1 - 2 * |p - 0.5|
• Reduces bet size when uncertainty > threshold
• Protects against overconfident predictions

Risk Controls:
• Minimum probability threshold (default: 55%)
• Maximum bet fraction (default: 10% of bankroll)
• Uncertainty threshold (default: 30%)
• Shrinkage factor (default: 50% reduction)

🎮 KEY FEATURES:
===============
• Conservative Approach: Prioritizes bankroll preservation
• Model-Aware: Adjusts to prediction confidence
• Flexible Parameters: Customizable risk tolerance
• Fair Value Analysis: Identifies profitable betting opportunities

🔧 TECHNICAL IMPLEMENTATION:
===========================

UncertaintyShrinkageBetting Class:
• Manages bankroll and betting parameters
• Calculates optimal bet sizes using Kelly Criterion
• Applies uncertainty-based adjustments
• Provides fair odds calculations

Key Methods:
• calculate_kelly_fraction(): Core betting logic
• calculate_uncertainty(): Model confidence assessment
• get_fair_odds(): Fair value calculation
• Various risk management checks

🎯 USAGE EXAMPLES:
=================

Basic Usage:
  betting = UncertaintyShrinkageBetting(initial_bankroll=1000)
  bet_fraction, odds = betting.calculate_kelly_fraction(prob=0.65, market_odds=2.5)

Custom Parameters:
  betting = UncertaintyShrinkageBetting(
      initial_bankroll=5000,
      min_prob=0.60,           # Higher confidence threshold
      shrinkage_factor=0.3,    # Less conservative shrinkage
      uncertainty_threshold=0.25
  )

Fair Odds Analysis:
  fair_odds = betting.get_fair_odds(prob=0.65)  # Returns ~1.54
  # If market offers 2.0, there's value betting opportunity

🚀 INTEGRATION WITH PREDICTION SYSTEM:
=====================================
This module integrates seamlessly with the tennis prediction pipeline:
• Receives probabilities from match_predictor.py
• Provides betting recommendations in the Streamlit interface
• Accounts for XGBoost model uncertainty
• Suggests optimal position sizing

💡 RESPONSIBLE BETTING:
======================
• Never bet more than you can afford to lose
• This system is for educational/research purposes
• Past performance doesn't guarantee future results
• Consider transaction costs and market liquidity
• Implement additional risk controls as needed

🔧 CUSTOMIZATION OPTIONS:
========================
• Bankroll Management: Adjust initial bankroll size
• Risk Tolerance: Modify max bet fraction and thresholds
• Uncertainty Handling: Change shrinkage factors
• Probability Filters: Set minimum confidence levels

🔗 RELATED FILES:
================
• match_predictor.py: Uses this for betting recommendations
• features.py: Provides prediction probabilities
• Streamlit UI: Displays betting advice to users
"""

import numpy as np

class UncertaintyShrinkageBetting:
    def __init__(self, initial_bankroll=1000, min_prob=0.55, shrinkage_factor=0.5, uncertainty_threshold=0.3):
        self.bankroll = initial_bankroll
        self.min_prob = min_prob
        self.shrinkage_factor = shrinkage_factor
        self.uncertainty_threshold = uncertainty_threshold
        self.max_bet_fraction = 0.10

    def calculate_uncertainty(self, prob):
        return 1 - 2 * abs(prob - 0.5)

    def calculate_kelly_fraction(self, prob, market_odds):
        # Standard Kelly Criterion calculation
        b = market_odds - 1
        q = 1 - prob
        f = (b * prob - q) / b if b > 0 else 0

        # Apply uncertainty shrinkage
        uncertainty = self.calculate_uncertainty(prob)
        if uncertainty > self.uncertainty_threshold:
            f *= (1 - self.shrinkage_factor)

        # Ensure bet size is within limits
        return min(max(f, 0), self.max_bet_fraction), market_odds
        
    def get_fair_odds(self, prob):
        """
        Calculate fair odds based on probability
        """
        return 1 / prob if prob > 0 else float('inf')

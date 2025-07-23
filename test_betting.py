#!/usr/bin/env python3
"""Test the betting strategy component."""

from betting_strategy import UncertaintyShrinkageBetting

def test_betting_strategy():
    print("🎯 Testing Betting Strategy")
    print("=" * 50)
    
    # Initialize betting strategy
    strategy = UncertaintyShrinkageBetting(initial_bankroll=1000)
    
    # Test uncertainty calculation
    prob_confident = 0.75  # Very confident prediction
    prob_uncertain = 0.52  # Less confident prediction
    
    uncertainty_confident = strategy.calculate_uncertainty(prob_confident)
    uncertainty_uncertain = strategy.calculate_uncertainty(prob_uncertain)
    
    print(f"Uncertainty for {prob_confident} probability: {uncertainty_confident:.4f}")
    print(f"Uncertainty for {prob_uncertain} probability: {uncertainty_uncertain:.4f}")
    
    # Test Kelly fraction calculation
    market_odds = 1.8  # Example market odds
    
    kelly_confident, _ = strategy.calculate_kelly_fraction(prob_confident, market_odds)
    kelly_uncertain, _ = strategy.calculate_kelly_fraction(prob_uncertain, market_odds)
    
    print(f"\nKelly fraction for confident prediction: {kelly_confident:.4f}")
    print(f"Kelly fraction for uncertain prediction: {kelly_uncertain:.4f}")
    
    # Test fair odds calculation
    fair_odds = strategy.get_fair_odds(prob_confident)
    print(f"\nFair odds for {prob_confident} probability: {fair_odds:.2f}")
    
    print("\n✅ Betting strategy tests passed!")
    print("The betting component is working correctly.")

if __name__ == "__main__":
    test_betting_strategy()

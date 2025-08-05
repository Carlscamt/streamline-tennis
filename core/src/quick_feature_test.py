#!/usr/bin/env python3
"""
Quick test to verify new features work before full feature engineering
"""

import pandas as pd
import sys
from features import FEATURES, ADDITIONAL_FEATURES, build_feature_vector

def test_new_features():
    """Test that new features can be computed with sample data"""
    print("🎾 TESTING NEW ADVANCED FEATURES")
    print("=" * 50)
    
    # Create sample player contexts (simulating what feature_engineering.py would provide)
    winner_context = {
        'career_win_pct': 0.65,
        'surface_win_pct': 0.70, 
        'recent_form': 0.75,
        'h2h_win_pct': 0.60,
        'elo_rating': 1800,
        'glicko2_rating': 1750,
        'glicko2_rd': 50,
        'glicko2_volatility': 0.06,
        'surface_dominance': 0.15,
        'surface_variability': 0.10,
        'fatigue_days': 7,
        'h2h_surface_win_pct': 0.65,
        'surface_adaptability': 0.85,
        'win_streak': 3,
        'loss_streak': 0,
        # New advanced features with default values
        'atp250_win_pct': 0.62,
        'atp500_win_pct': 0.68,
        'masters1000_win_pct': 0.58,
        'grand_slam_win_pct': 0.55,
        'hard_court_win_pct': 0.72,
        'clay_court_win_pct': 0.45,
        'grass_court_win_pct': 0.68,
        'early_round_win_pct': 0.75,
        'late_round_win_pct': 0.60,
        'quarterfinal_win_pct': 0.55,
        'big_match_experience': 0.25,
        'surface_transition_perf': 0.68,
        'recent_hard_court_form': 0.70,
        'recent_clay_court_form': 0.50,
        'recent_grass_court_form': 0.75,
        'recent_big_tournament_form': 0.65,
        'clutch_performance': 0.72,
        'outdoor_preference': 0.05
    }
    
    loser_context = {
        'career_win_pct': 0.58,
        'surface_win_pct': 0.55,
        'recent_form': 0.45,
        'h2h_win_pct': 0.40,
        'elo_rating': 1650,
        'glicko2_rating': 1600, 
        'glicko2_rd': 75,
        'glicko2_volatility': 0.08,
        'surface_dominance': 0.05,
        'surface_variability': 0.20,
        'fatigue_days': 3,
        'h2h_surface_win_pct': 0.35,
        'surface_adaptability': 0.75,
        'win_streak': 1,
        'loss_streak': 2,
        # New advanced features
        'atp250_win_pct': 0.60,
        'atp500_win_pct': 0.55,
        'masters1000_win_pct': 0.50,
        'grand_slam_win_pct': 0.45,
        'hard_court_win_pct': 0.58,
        'clay_court_win_pct': 0.52,
        'grass_court_win_pct': 0.48,
        'early_round_win_pct': 0.65,
        'late_round_win_pct': 0.45,
        'quarterfinal_win_pct': 0.42,
        'big_match_experience': 0.15,
        'surface_transition_perf': 0.58,
        'recent_hard_court_form': 0.50,
        'recent_clay_court_form': 0.55,
        'recent_grass_court_form': 0.45,
        'recent_big_tournament_form': 0.48,
        'clutch_performance': 0.55,
        'outdoor_preference': -0.02
    }
    
    # Test all features
    ALL_FEATURES = FEATURES + ADDITIONAL_FEATURES
    print(f"Testing {len(ALL_FEATURES)} total features:")
    print(f"- Core features: {len(FEATURES)}")
    print(f"- Additional features: {len(ADDITIONAL_FEATURES)}")
    print()
    
    try:
        # Test feature vector building with ALL features
        feature_vector = build_feature_vector(
            winner_context=winner_context,
            loser_context=loser_context,
            feature_list=ALL_FEATURES
        )
        
        print("✅ SUCCESS: All features computed successfully!")
        print()
        print("Feature values preview:")
        for i, (feature_name, value) in enumerate(feature_vector.items()):
            if feature_name != 'Win':  # Skip the label
                print(f"  {feature_name:<35} {value:>8.4f}")
                if i >= 15:  # Show first 15 features
                    remaining = len(ALL_FEATURES) - 15
                    if remaining > 0:
                        print(f"  ... and {remaining} more features")
                    break
        
        print()
        print(f"🎯 READY FOR FEATURE ENGINEERING!")
        print(f"All {len(ALL_FEATURES)} features are properly implemented and tested.")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print()
        print("This indicates a problem with feature implementation.")
        return False

if __name__ == "__main__":
    success = test_new_features()
    sys.exit(0 if success else 1)

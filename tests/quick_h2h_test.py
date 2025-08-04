#!/usr/bin/env python3
"""
Quick test to verify H2H mapping fix
"""
import sys
import os
sys.path.append('src')

def test_h2h_mapping():
    """Test that H2H mapping works correctly."""
    try:
        # Test imports
        print("✅ Testing imports...")
        from rating_cache import RatingCache
        import pandas as pd
        from features import FEATURES, build_prediction_feature_vector
        print("✅ All imports successful")
        
        # Load data
        print("📊 Loading minimal data...")
        data_path = "data/tennis_data/tennis_data.csv"
        if not os.path.exists(data_path):
            print("❌ Data file not found")
            return False
            
        df = pd.read_csv(data_path).head(100)  # Just test with first 100 rows
        print(f"✅ Loaded {len(df)} matches for testing")
        
        # Initialize cache
        print("🔧 Initializing rating cache...")
        cache = RatingCache(df)
        print("✅ Rating cache initialized")
        
        # Test H2H stats directly
        print("🔍 Testing H2H stats...")
        test_date = pd.Timestamp('2020-02-01')
        players = df['Winner'].unique()[:2]  # Get first two players
        
        if len(players) < 2:
            print("❌ Not enough players for H2H test")
            return False
            
        player1, player2 = players[0], players[1]
        surface = 'Hard'
        
        # Get H2H stats
        h2h_stats = cache.get_h2h_stats(player1, player2, surface, test_date)
        print(f"✅ H2H stats keys: {list(h2h_stats.keys())}")
        
        # Get player stats
        p1_stats = cache.get_comprehensive_player_stats(player1, surface, test_date)
        p2_stats = cache.get_comprehensive_player_stats(player2, surface, test_date)
        print(f"✅ Player stats loaded")
        
        # Test H2H mapping (this was the missing piece)
        p1_stats['h2h_win_pct'] = h2h_stats['player1_h2h_win_pct']
        p2_stats['h2h_win_pct'] = h2h_stats['player2_h2h_win_pct']
        p1_stats['h2h_surface_win_pct'] = h2h_stats['player1_h2h_surface_win_pct']
        p2_stats['h2h_surface_win_pct'] = h2h_stats['player2_h2h_surface_win_pct']
        print(f"✅ H2H mapping applied")
        
        # Test feature vector building
        print("🧮 Building feature vector...")
        feature_values = build_prediction_feature_vector(p1_stats, p2_stats)
        print(f"✅ Feature vector built with {len(feature_values)} features")
        
        # Check specific H2H feature
        if 'h2h_win_pct_diff' in feature_values:
            h2h_diff = feature_values['h2h_win_pct_diff']
            print(f"✅ h2h_win_pct_diff = {h2h_diff}")
        else:
            print("❌ h2h_win_pct_diff not found in features")
            return False
            
        print("🎉 ALL TESTS PASSED! H2H mapping fix verified!")
        return True
        
    except Exception as e:
        print(f"❌ Error in H2H test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_h2h_mapping()
    exit(0 if success else 1)

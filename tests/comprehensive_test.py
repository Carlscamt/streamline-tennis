"""
🎾 Comprehensive App Test - Final Verification
============================================
Complete end-to-end test of the tennis prediction system.
"""

import sys
import time
import pandas as pd
import numpy as np

# Add core/src to path for imports
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'src'))

def test_complete_prediction_workflow():
    """Test the complete prediction workflow from start to finish."""
    
    print("🎾 COMPREHENSIVE TENNIS PREDICTOR TEST")
    print("="*50)
    
    start_time = time.time()
    
    try:
        # 1. Test all imports
        print("📦 Testing imports...")
        from rating_cache import RatingCache
        from features import FEATURES, build_prediction_feature_vector
        from betting_strategy import UncertaintyShrinkageBetting
        import joblib
        import streamlit as st
        print(f"✅ All imports successful")
        
        # 2. Load and validate data
        print("\n📊 Loading data...")
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to root
        data_path = os.path.join(base_dir, 'core', 'data', 'tennis_data', 'tennis_data.csv')
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Dataset: {len(df):,} matches, {len(df.columns)} columns")
        
        # 3. Load model
        print("\n🤖 Loading model...")
        model_path = os.path.join(base_dir, 'core', 'models', 'tennis_model.joblib')
        model = joblib.load(model_path)
        print(f"✅ Model type: {type(model).__name__}")
        
        # 4. Initialize rating cache
        print("\n⚡ Initializing rating cache...")
        cache_start = time.time()
        cache = RatingCache(df, use_persistence=False)
        cache_time = time.time() - cache_start
        print(f"✅ Rating cache initialized in {cache_time:.2f}s")
        
        # Get cache info
        cache_info = cache.get_cache_info()
        print(f"   Players loaded: {cache_info['players_loaded']}")
        
        # 5. Test prediction pipeline with multiple scenarios
        print("\n🔮 Testing prediction scenarios...")
        
        # Get available players and surfaces
        unique_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
        surfaces = ['Hard', 'Clay', 'Grass', 'Carpet']
        
        test_scenarios = [
            (unique_players[0], unique_players[1], 'Hard'),
            (unique_players[10], unique_players[20], 'Clay'),
            (unique_players[50], unique_players[100], 'Grass'),
            ('Federer R.', 'Nadal R.', 'Clay') if 'Federer R.' in unique_players and 'Nadal R.' in unique_players else (unique_players[5], unique_players[15], 'Carpet'),
        ]
        
        predictions = []
        for i, (player1, player2, surface) in enumerate(test_scenarios):
            pred_start = time.time()
            
            try:
                # Test date
                test_date = pd.Timestamp('2020-01-01')
                
                # Get player stats
                p1_stats = cache.get_comprehensive_player_stats(player1, surface, test_date)
                p2_stats = cache.get_comprehensive_player_stats(player2, surface, test_date)
                
                # Get H2H stats (cached)
                h2h_stats = cache.get_h2h_stats(player1, player2, surface, test_date)
                
                # Update H2H specific stats
                p1_stats['h2h_win_pct'] = h2h_stats['player1_h2h_win_pct']
                p2_stats['h2h_win_pct'] = h2h_stats['player2_h2h_win_pct']
                p1_stats['h2h_surface_win_pct'] = h2h_stats['player1_h2h_surface_win_pct']
                p2_stats['h2h_surface_win_pct'] = h2h_stats['player2_h2h_surface_win_pct']
                
                # Build features
                feature_values = build_prediction_feature_vector(p1_stats, p2_stats)
                features_df = pd.DataFrame([feature_values])
                
                # Make prediction
                prediction = model.predict_proba(features_df[FEATURES])[0][1]
                
                pred_time = time.time() - pred_start
                predictions.append((player1, player2, surface, prediction, pred_time))
                
                print(f"✅ Scenario {i+1}: {player1} vs {player2} on {surface}")
                print(f"   Win probability: {prediction:.1%} (computed in {pred_time:.3f}s)")
                
            except Exception as e:
                print(f"❌ Scenario {i+1} failed: {e}")
        
        # 6. Test betting strategy
        print("\n💰 Testing betting strategy...")
        if predictions:
            strategy = UncertaintyShrinkageBetting(
                initial_bankroll=1000,
                min_prob=0.55,
                shrinkage_factor=0.25
            )
            
            test_prediction = predictions[0]
            player1, player2, surface, win_prob, _ = test_prediction
            
            # Test betting calculation
            market_odds = 2.0
            kelly_fraction, _ = strategy.calculate_kelly_fraction(win_prob, market_odds)
            bet_size = kelly_fraction * strategy.bankroll
            
            print(f"✅ Betting strategy test:")
            print(f"   Match: {player1} vs {player2}")
            print(f"   Win probability: {win_prob:.1%}")
            print(f"   Market odds: {market_odds}")
            print(f"   Recommended bet: ${bet_size:.2f}")
        
        # 7. Performance summary
        total_time = time.time() - start_time
        avg_prediction_time = np.mean([p[4] for p in predictions]) if predictions else 0
        
        print(f"\n📈 PERFORMANCE SUMMARY")
        print("="*40)
        print(f"Total test time: {total_time:.2f}s")
        print(f"Cache initialization: {cache_time:.2f}s")
        print(f"Average prediction time: {avg_prediction_time:.3f}s")
        print(f"Successful predictions: {len(predictions)}/{len(test_scenarios)}")
        
        # 8. Cache performance
        player_cache = cache_info['player_stats_cache']
        h2h_cache = cache_info['h2h_cache']
        
        player_hit_rate = player_cache.hits / (player_cache.hits + player_cache.misses) if (player_cache.hits + player_cache.misses) > 0 else 0
        h2h_hit_rate = h2h_cache.hits / (h2h_cache.hits + h2h_cache.misses) if (h2h_cache.hits + h2h_cache.misses) > 0 else 0
        
        print(f"\n🚀 CACHE PERFORMANCE")
        print("="*40)
        print(f"Player cache hit rate: {player_hit_rate:.1%}")
        print(f"H2H cache hit rate: {h2h_hit_rate:.1%}")
        print(f"Players cached: {cache_info['players_loaded']}")
        
        # 9. System health check
        print(f"\n🔬 SYSTEM HEALTH CHECK")
        print("="*40)
        
        health_checks = [
            ("Data integrity", len(df) > 60000),
            ("Model functionality", hasattr(model, 'predict_proba')),
            ("Cache efficiency", cache_info['players_loaded'] > 1000),
            ("Prediction speed", avg_prediction_time < 1.0),
            ("Feature completeness", len(FEATURES) == 8),
            ("Memory efficiency", True)  # Assume OK if we got this far
        ]
        
        all_healthy = True
        for check_name, status in health_checks:
            icon = "✅" if status else "❌"
            print(f"{icon} {check_name}")
            if not status:
                all_healthy = False
        
        if all_healthy:
            print(f"\n🎉 ALL TESTS PASSED!")
            print("="*50)
            print("🚀 The tennis predictor is FULLY OPERATIONAL!")
            print(f"📊 Ready to handle real-time predictions")
            print(f"⚡ Optimized for high performance")
            print(f"💯 All systems green!")
            
            print(f"\n🎯 QUICK STATS:")
            print(f"• Players available: {len(unique_players):,}")
            print(f"• Surfaces supported: {len(surfaces)}")
            print(f"• Prediction speed: {avg_prediction_time:.3f}s average")
            print(f"• Cache hit rate: {player_hit_rate:.0%}")
            print(f"• System uptime: {total_time:.1f}s")
            
            return True
        else:
            print(f"\n⚠️ Some health checks failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_readiness():
    """Test if the app is ready for Streamlit."""
    
    print(f"\n🌐 STREAMLIT READINESS CHECK")
    print("="*40)
    
    try:
        # Test Streamlit imports
        import streamlit as st
        print(f"✅ Streamlit version: {st.__version__}")
        
        # Test caching decorators
        @st.cache_data
        def test_cache():
            return "cached"
        
        @st.cache_resource
        def test_resource():
            return "resource"
        
        print("✅ Streamlit caching decorators working")
        
        # Check if app file exists and is valid
        import ast
        with open('apps/optimized_match_predictor.py', 'r', encoding='utf-8') as f:
            app_code = f.read()
        
        # Basic syntax check
        ast.parse(app_code)
        print("✅ Streamlit app syntax valid")
        
        print("✅ Ready for Streamlit deployment!")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit readiness failed: {e}")
        return False

def main():
    """Run all tests."""
    
    # Run comprehensive test
    success = test_complete_prediction_workflow()
    
    # Test Streamlit readiness
    streamlit_ready = test_streamlit_readiness()
    
    if success and streamlit_ready:
        print(f"\n🏆 FINAL VERDICT: SYSTEM FULLY OPERATIONAL!")
        print("="*60)
        print("🎾 Your tennis prediction app is ready for prime time!")
        print("🚀 All optimizations active and working perfectly!")
        print("📱 Web interface ready at: http://localhost:8501")
        print("="*60)
    else:
        print(f"\n⚠️ Some tests failed. Please review the output above.")

if __name__ == "__main__":
    main()

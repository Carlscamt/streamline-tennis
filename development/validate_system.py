"""
🎾 Quick Validation: Tennis Predictor System
==========================================
Validates that all components are working before launching the full app.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

def validate_system():
    """Validate all system components."""
    
    print("🎾 OPTIMIZED TENNIS PREDICTOR VALIDATION")
    print("="*50)
    
    # 1. Test Data Loading
    print("\n📊 Testing data loading...")
    try:
        df = pd.read_csv('data/tennis_data/tennis_data.csv', low_memory=False)
        print(f"✅ Raw data loaded: {len(df):,} matches")
        print(f"   Columns: {list(df.columns[:10])}...")
        
        # Get sample players and surfaces
        unique_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
        surfaces = sorted(df['Surface'].unique())
        
        print(f"   Players: {len(unique_players)} (e.g., {unique_players[:5]})")
        print(f"   Surfaces: {surfaces}")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # 2. Test Model Loading
    print("\n🤖 Testing model loading...")
    try:
        import joblib
        model = joblib.load('models/tennis_model.joblib')
        print(f"✅ Model loaded: {type(model)}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # 3. Test Imports
    print("\n📦 Testing module imports...")
    try:
        from features import FEATURES, build_prediction_feature_vector
        print(f"✅ Features module: {len(FEATURES)} features")
        
        from rating_cache import RatingCache
        print("✅ Rating cache module imported")
        
        from betting_strategy import UncertaintyShrinkageBetting
        print("✅ Betting strategy module imported")
        
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        return False
    
    # 4. Test Rating Cache Initialization
    print("\n⚡ Testing rating cache initialization...")
    try:
        cache = RatingCache(df, use_persistence=False)
        cache_info = cache.get_cache_info()
        print(f"✅ Rating cache initialized: {cache_info['players_loaded']} players loaded")
        
        # Test a quick lookup
        test_player = unique_players[0]
        test_surface = surfaces[0]
        test_date = pd.Timestamp('2020-01-01')
        
        stats = cache.get_comprehensive_player_stats(test_player, test_surface, test_date)
        print(f"✅ Sample stats for {test_player}: ELO={stats['elo_rating']:.0f}")
        
    except Exception as e:
        print(f"❌ Rating cache failed: {e}")
        return False
    
    # 5. Test Prediction Pipeline
    print("\n🔮 Testing prediction pipeline...")
    try:
        player1, player2 = unique_players[0], unique_players[1]
        surface = surfaces[0]
        test_date = pd.Timestamp('2020-01-01')
        
        # Get stats for both players
        p1_stats = cache.get_comprehensive_player_stats(player1, surface, test_date)
        p2_stats = cache.get_comprehensive_player_stats(player2, surface, test_date)
        
        # Build feature vector
        feature_values = build_prediction_feature_vector(p1_stats, p2_stats)
        features_df = pd.DataFrame([feature_values])
        
        # Make prediction
        prediction = model.predict_proba(features_df[FEATURES])[0][1]
        
        print(f"✅ Test prediction: {player1} vs {player2} on {surface}")
        print(f"   {player1} win probability: {prediction:.1%}")
        
    except Exception as e:
        print(f"❌ Prediction pipeline failed: {e}")
        return False
    
    # Success!
    print("\n" + "="*50)
    print("🎉 ALL VALIDATIONS PASSED!")
    print("🚀 System is ready for launch!")
    print("\nNext steps:")
    print("1. Run: streamlit run optimized_match_predictor.py")
    print("2. Open browser to: http://localhost:8501")
    print("3. Test the interactive prediction interface")
    print("="*50)
    
    return True

if __name__ == "__main__":
    validate_system()

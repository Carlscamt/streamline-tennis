"""
🎾 Final App Test - Post Error Fix
=================================
Tests the app after fixing import errors.
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_app_components():
    """Test all app components work together."""
    
    print("🎾 TESTING FIXED APP COMPONENTS")
    print("="*45)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from rating_cache import RatingCache
        from features import FEATURES, build_prediction_feature_vector
        from betting_strategy import UncertaintyShrinkageBetting
        import pandas as pd
        import joblib
        import streamlit as st
        print("✅ All imports successful!")
        
        # Test data loading
        print("\n📊 Testing data loading...")
        df = pd.read_csv('data/tennis_data/tennis_data.csv', low_memory=False)
        print(f"✅ Data loaded: {len(df):,} matches")
        
        # Test model loading
        print("\n🤖 Testing model loading...")
        model = joblib.load('models/tennis_model.joblib')
        print("✅ Model loaded successfully")
        
        # Test rating cache
        print("\n⚡ Testing rating cache...")
        cache = RatingCache(df, use_persistence=False)
        cache_info = cache.get_cache_info()
        print(f"✅ Rating cache: {cache_info['players_loaded']} players loaded")
        
        # Test prediction pipeline
        print("\n🔮 Testing prediction pipeline...")
        unique_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
        player1, player2 = unique_players[0], unique_players[1]
        surface = 'Hard'
        test_date = pd.Timestamp('2020-01-01')
        
        # Get player stats
        p1_stats = cache.get_comprehensive_player_stats(player1, surface, test_date)
        p2_stats = cache.get_comprehensive_player_stats(player2, surface, test_date)
        
        # Build features
        feature_values = build_prediction_feature_vector(p1_stats, p2_stats)
        features_df = pd.DataFrame([feature_values])
        
        # Make prediction
        prediction = model.predict_proba(features_df[FEATURES])[0][1]
        
        print(f"✅ Test prediction successful!")
        print(f"   Match: {player1} vs {player2}")
        print(f"   Surface: {surface}")
        print(f"   {player1} win probability: {prediction:.1%}")
        
        print("\n🎉 ALL COMPONENTS WORKING!")
        print("🚀 App is ready to launch!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def provide_launch_instructions():
    """Provide instructions for launching the app."""
    
    print("\n📋 LAUNCH INSTRUCTIONS")
    print("="*45)
    print("1. Open a new terminal/command prompt")
    print("2. Navigate to the project directory:")
    print("   cd \"c:\\Users\\Carlos\\Documents\\ML\\Tennis\\streamline-tennis\"")
    print("3. Activate the virtual environment:")
    print("   ../.venv/Scripts/python.exe -m streamlit run optimized_match_predictor.py")
    print("4. Open your browser to: http://localhost:8501")
    print("\n🎯 Expected Features:")
    print("• Player selection (1,767 players available)")
    print("• Surface selection (Hard, Clay, Grass, Carpet)")
    print("• Instant predictions with detailed analysis")
    print("• Betting recommendations")
    print("• Performance metrics")

if __name__ == "__main__":
    success = test_app_components()
    
    if success:
        provide_launch_instructions()
    else:
        print("\n❌ App still has issues. Please check the errors above.")

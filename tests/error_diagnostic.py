"""
🔍 Tennis Predictor Error Diagnostic Tool
========================================
Comprehensive error checking for the optimized tennis match predictor.
"""

import sys
import traceback
import pandas as pd
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def check_file_existence():
    """Check if all required files exist."""
    print("📁 CHECKING FILE EXISTENCE")
    print("="*40)
    
    required_files = [
        'data/tennis_data/tennis_data.csv',
        'models/tennis_model.joblib',
        'src/features.py',
        'src/rating_cache.py',
        'src/betting_strategy.py',
        'optimized_match_predictor.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"❌ MISSING: {file_path}")
            all_exist = False
    
    return all_exist

def check_data_integrity():
    """Check data file integrity."""
    print("\n📊 CHECKING DATA INTEGRITY")
    print("="*40)
    
    try:
        # Load main dataset
        df = pd.read_csv('data/tennis_data/tennis_data.csv', low_memory=False)
        print(f"✅ Dataset loaded: {len(df):,} matches")
        
        # Check required columns
        required_cols = ['Date', 'Winner', 'Loser', 'Surface']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        else:
            print(f"✅ All required columns present: {required_cols}")
        
        # Check for null values in critical columns
        for col in required_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"⚠️  {col} has {null_count:,} null values")
            else:
                print(f"✅ {col}: No null values")
        
        # Check date range
        df['Date'] = pd.to_datetime(df['Date'])
        date_range = df['Date'].max() - df['Date'].min()
        print(f"✅ Date range: {df['Date'].min().date()} to {df['Date'].max().date()} ({date_range.days} days)")
        
        # Check unique values
        unique_players = set(df['Winner'].unique()) | set(df['Loser'].unique())
        surfaces = df['Surface'].unique()
        print(f"✅ Unique players: {len(unique_players):,}")
        print(f"✅ Surfaces: {list(surfaces)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data integrity check failed: {e}")
        traceback.print_exc()
        return False

def check_model_loading():
    """Check model loading and compatibility."""
    print("\n🤖 CHECKING MODEL LOADING")
    print("="*40)
    
    try:
        import joblib
        model = joblib.load('models/tennis_model.joblib')
        print(f"✅ Model loaded: {type(model)}")
        
        # Check if model has required methods
        required_methods = ['predict', 'predict_proba']
        for method in required_methods:
            if hasattr(model, method):
                print(f"✅ Model has {method} method")
            else:
                print(f"❌ Model missing {method} method")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return False

def check_module_imports():
    """Check if all custom modules can be imported."""
    print("\n📦 CHECKING MODULE IMPORTS")
    print("="*40)
    
    modules_to_test = [
        ('features', ['FEATURES', 'build_prediction_feature_vector']),
        ('rating_cache', ['RatingCache']),
        ('betting_strategy', ['UncertaintyShrinkageBetting'])
    ]
    
    all_imported = True
    
    for module_name, expected_attrs in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"✅ {module_name} imported successfully")
            
            # Check expected attributes
            for attr in expected_attrs:
                if hasattr(module, attr):
                    print(f"  ✅ {attr} available")
                else:
                    print(f"  ❌ {attr} missing")
                    all_imported = False
                    
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            all_imported = False
        except Exception as e:
            print(f"❌ Error with {module_name}: {e}")
            all_imported = False
    
    return all_imported

def check_rating_cache_functionality():
    """Test rating cache functionality."""
    print("\n⚡ CHECKING RATING CACHE FUNCTIONALITY")
    print("="*40)
    
    try:
        # Load data
        df = pd.read_csv('data/tennis_data/tennis_data.csv', low_memory=False)
        
        # Import and create rating cache
        from rating_cache import RatingCache
        cache = RatingCache(df, use_persistence=False)
        
        cache_info = cache.get_cache_info()
        print(f"✅ Rating cache initialized: {cache_info['players_loaded']} players")
        
        # Test player lookup
        unique_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
        test_player = unique_players[0]
        test_surface = df['Surface'].iloc[0]
        test_date = pd.Timestamp('2020-01-01')
        
        stats = cache.get_comprehensive_player_stats(test_player, test_surface, test_date)
        print(f"✅ Player stats retrieved for {test_player}")
        print(f"  ELO Rating: {stats['elo_rating']:.0f}")
        print(f"  Career Win %: {stats['career_win_pct']:.1%}")
        
        # Test H2H lookup
        test_player2 = unique_players[1]
        h2h_stats = cache.get_h2h_stats(test_player, test_player2, test_surface, test_date)
        print(f"✅ H2H stats retrieved: {test_player} vs {test_player2}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rating cache functionality failed: {e}")
        traceback.print_exc()
        return False

def check_prediction_pipeline():
    """Test the complete prediction pipeline."""
    print("\n🔮 CHECKING PREDICTION PIPELINE")
    print("="*40)
    
    try:
        # Load components
        import joblib
        df = pd.read_csv('data/tennis_data/tennis_data.csv', low_memory=False)
        model = joblib.load('models/tennis_model.joblib')
        
        from rating_cache import RatingCache
        from features import FEATURES, build_prediction_feature_vector
        
        cache = RatingCache(df, use_persistence=False)
        
        # Test prediction
        unique_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
        player1, player2 = unique_players[0], unique_players[1]
        surface = df['Surface'].iloc[0]
        test_date = pd.Timestamp('2020-01-01')
        
        print(f"Testing prediction: {player1} vs {player2} on {surface}")
        
        # Get player stats
        p1_stats = cache.get_comprehensive_player_stats(player1, surface, test_date)
        p2_stats = cache.get_comprehensive_player_stats(player2, surface, test_date)
        
        # Build features
        feature_values = build_prediction_feature_vector(p1_stats, p2_stats)
        features_df = pd.DataFrame([feature_values])
        
        print(f"✅ Features computed: {len(FEATURES)} features")
        
        # Make prediction
        prediction = model.predict_proba(features_df[FEATURES])[0][1]
        print(f"✅ Prediction successful: {player1} win probability = {prediction:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction pipeline failed: {e}")
        traceback.print_exc()
        return False

def check_streamlit_compatibility():
    """Check Streamlit-specific functionality."""
    print("\n🌐 CHECKING STREAMLIT COMPATIBILITY")
    print("="*40)
    
    try:
        import streamlit as st
        print(f"✅ Streamlit version: {st.__version__}")
        
        # Test caching decorators
        @st.cache_data
        def test_cache_data():
            return "cached data"
        
        @st.cache_resource  
        def test_cache_resource():
            return "cached resource"
        
        print("✅ Streamlit caching decorators working")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit compatibility check failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive error checking."""
    print("🔍 TENNIS PREDICTOR ERROR DIAGNOSTIC")
    print("="*50)
    print("Running comprehensive error checks...\n")
    
    # Run all checks
    checks = [
        ("File Existence", check_file_existence),
        ("Data Integrity", check_data_integrity),
        ("Model Loading", check_model_loading),
        ("Module Imports", check_module_imports),
        ("Rating Cache", check_rating_cache_functionality),
        ("Prediction Pipeline", check_prediction_pipeline),
        ("Streamlit Compatibility", check_streamlit_compatibility)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check crashed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*50)
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<25} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("The tennis predictor should be working correctly.")
        print("\nIf you're still having issues:")
        print("1. Try refreshing the browser at http://localhost:8501")
        print("2. Check if the Streamlit process is running")
        print("3. Look for error messages in the terminal")
    else:
        print("⚠️  SOME CHECKS FAILED!")
        print("Please fix the failed checks before using the app.")
        print("\nCommon solutions:")
        print("1. Ensure all data files are present")
        print("2. Reinstall dependencies: pip install -r requirements.txt")
        print("3. Check file permissions")
    
    return all_passed

if __name__ == "__main__":
    main()

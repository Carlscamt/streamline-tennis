"""
Simple test script to verify all components work before running the full app.
"""

import sys
import os
import pandas as pd

# Add src to path
sys.path.append('src')

def test_imports():
    """Test all required imports."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import joblib
        print("✅ Joblib imported")
    except ImportError as e:
        print(f"❌ Joblib import failed: {e}")
        return False
    
    try:
        from src.rating_cache import RatingCache
        print("✅ RatingCache imported")
    except ImportError as e:
        print(f"❌ RatingCache import failed: {e}")
        return False
    
    try:
        from src.features import FEATURES, build_prediction_feature_vector
        print("✅ Features imported")
        print(f"   Available features: {len(FEATURES)}")
        print(f"   Features: {FEATURES}")
    except ImportError as e:
        print(f"❌ Features import failed: {e}")
        return False
    
    try:
        from src.betting_strategy import UncertaintyShrinkageBetting
        print("✅ Betting strategy imported")
    except ImportError as e:
        print(f"❌ Betting strategy import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test model loading."""
    print("\n🤖 Testing model loading...")
    
    try:
        import joblib
        model = joblib.load('models/tennis_model.joblib')
        print("✅ Model loaded successfully")
        print(f"   Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_data_loading():
    """Test data loading."""
    print("\n📊 Testing data loading...")
    
    data_paths = [
        'data/tennis_data/tennis_data.csv',
        'tennis_data/tennis_data.csv',
        'data/tennis_data.csv',
        'tennis_data.csv'
    ]
    
    for path in data_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                print(f"✅ Data loaded from {path}")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns[:10])}...")  # Show first 10 columns
                return df, True
        except Exception as e:
            print(f"❌ Failed to load {path}: {e}")
    
    print("❌ No data file found")
    return None, False

def test_rating_cache():
    """Test rating cache initialization."""
    print("\n⚡ Testing rating cache...")
    
    try:
        # Load test data
        df, success = test_data_loading()
        if not success:
            return False
        
        # Import and test rating cache
        from src.rating_cache import RatingCache
        cache = RatingCache(df, use_persistence=False)
        print("✅ Rating cache initialized")
        
        cache_info = cache.get_cache_info()
        print(f"   Players loaded: {cache_info['players_loaded']}")
        return True
        
    except Exception as e:
        print(f"❌ Rating cache test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎾 TENNIS PREDICTOR COMPONENT TEST")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Model Loading", test_model_loading), 
        ("Data Loading", test_data_loading),
        ("Rating Cache", test_rating_cache)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📋 TEST RESULTS SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to run Streamlit app!")
        print("\nNext step: streamlit run optimized_match_predictor.py")
    else:
        print("⚠️  Some tests failed. Please fix issues before running the app.")
    
    return all_passed

if __name__ == "__main__":
    main()

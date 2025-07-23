#!/usr/bin/env python3
"""
Backup Verification Script for Tennis Predictor - Glicko-2 Implementation
Verifies that all files are present and functional
"""

import os
import sys
from pathlib import Path
import joblib
import pandas as pd

def check_file(file_path: str, description: str) -> bool:
    """Check if a file exists and return status."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def verify_backup():
    """Verify the backup is complete and functional."""
    print("🎾 Tennis Predictor Backup Verification")
    print("=" * 45)
    
    all_good = True
    
    # Core files
    files_to_check = [
        ("feature_engineering_fixed_no_leakage.py", "Feature Engineering Script"),
        ("train_model_no_leakage.py", "Model Training Script"),
        ("match_predictor.py", "Web Application"),
        ("test_glicko2.py", "Test Suite"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Documentation"),
        ("setup.py", "Setup Script"),
    ]
    
    print("\n📄 Core Files:")
    for file_path, description in files_to_check:
        if not check_file(file_path, description):
            all_good = False
    
    # Model files
    print("\n🤖 Model Files:")
    model_files = [
        ("models/tennis_model_no_leakage.joblib", "Trained XGBoost Model"),
        ("models/surface_encoder.joblib", "Surface Label Encoder"),
    ]
    
    for file_path, description in model_files:
        if not check_file(file_path, description):
            all_good = False
    
    # Data files
    print("\n📊 Data Files:")
    data_files = [
        ("data/tennis_data_features_no_leakage.csv", "Feature Dataset"),
        ("data/feature_importance_no_leakage.csv", "Feature Importance"),
    ]
    
    for file_path, description in data_files:
        if not check_file(file_path, description):
            all_good = False
    
    # Documentation
    print("\n📖 Documentation:")
    doc_files = [
        ("documentation/DATA_LEAKAGE_ANALYSIS.md", "Data Leakage Analysis"),
        ("documentation/PROJECT_CONTEXT.md", "Project Context"),
    ]
    
    for file_path, description in doc_files:
        if not check_file(file_path, description):
            all_good = False
    
    # Verify data integrity
    print("\n🔍 Data Integrity Check:")
    try:
        if os.path.exists("data/tennis_data_features_no_leakage.csv"):
            df = pd.read_csv("data/tennis_data_features_no_leakage.csv")
            expected_shape = (65715, 14)
            if df.shape == expected_shape:
                print(f"✅ Dataset shape: {df.shape} (expected: {expected_shape})")
            else:
                print(f"❌ Dataset shape: {df.shape} (expected: {expected_shape})")
                all_good = False
            
            expected_columns = [
                'Date', 'Winner', 'Loser', 'Surface', 'Win', 'career_win_pct_diff',
                'surface_win_pct_diff', 'recent_form_diff', 'h2h_win_pct_diff',
                'ranking_diff', 'elo_rating_diff', 'glicko2_rating_diff',
                'glicko2_rd_diff', 'glicko2_volatility_diff'
            ]
            if all(col in df.columns for col in expected_columns):
                print(f"✅ All expected columns present ({len(expected_columns)} columns)")
            else:
                missing = [col for col in expected_columns if col not in df.columns]
                print(f"❌ Missing columns: {missing}")
                all_good = False
                
    except Exception as e:
        print(f"❌ Data integrity check failed: {e}")
        all_good = False
    
    # Verify model integrity
    print("\n🤖 Model Integrity Check:")
    try:
        if os.path.exists("models/tennis_model_no_leakage.joblib"):
            model = joblib.load("models/tennis_model_no_leakage.joblib")
            print(f"✅ Model loaded successfully: {type(model).__name__}")
            
            # Check if it has the expected number of features
            if hasattr(model, 'n_features_in_'):
                expected_features = 9
                if model.n_features_in_ == expected_features:
                    print(f"✅ Model expects {model.n_features_in_} features (correct)")
                else:
                    print(f"❌ Model expects {model.n_features_in_} features (expected: {expected_features})")
                    all_good = False
        
    except Exception as e:
        print(f"❌ Model integrity check failed: {e}")
        all_good = False
    
    # Final result
    print("\n" + "=" * 45)
    if all_good:
        print("🎉 BACKUP VERIFICATION SUCCESSFUL!")
        print("   All files are present and appear to be functional.")
        print("   You can deploy this backup anywhere and run:")
        print("   python setup.py")
        print("   streamlit run match_predictor.py")
    else:
        print("❌ BACKUP VERIFICATION FAILED!")
        print("   Some files are missing or corrupted.")
        print("   Please check the issues listed above.")
    
    return all_good

if __name__ == "__main__":
    success = verify_backup()
    sys.exit(0 if success else 1)

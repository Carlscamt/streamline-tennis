#!/usr/bin/env python3
"""
Setup script for Tennis Predictor - Glicko-2 Implementation
Automates the setup process for the backup directory
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🎾 Tennis Predictor - Glicko-2 Implementation Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("feature_engineering_fixed_no_leakage.py"):
        print("❌ Please run this script from the backup directory containing the implementation files")
        sys.exit(1)
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("⚠️  Dependency installation failed. Please install manually:")
        print("pip install -r requirements.txt")
        return False
    
    # Check if model and data files exist
    model_exists = os.path.exists("models/tennis_model_no_leakage.joblib")
    data_exists = os.path.exists("data/tennis_data_features_no_leakage.csv")
    
    if not data_exists:
        print("📊 Generating feature dataset...")
        if not run_command("python feature_engineering_fixed_no_leakage.py", "Generating features"):
            return False
    
    if not model_exists:
        print("🤖 Training model...")
        if not run_command("python train_model_no_leakage.py", "Training model"):
            return False
    
    # Run tests
    if not run_command("python test_glicko2.py", "Running Glicko-2 tests"):
        print("⚠️  Tests failed, but setup can continue")
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 To start the web application, run:")
    print("streamlit run match_predictor.py")
    print("\n📖 For more information, see README.md")
    
    return True

if __name__ == "__main__":
    main()

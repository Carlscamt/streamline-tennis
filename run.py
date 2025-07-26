#!/usr/bin/env python3
"""
🎾 Tennis Match Prediction - Quick Start Script
==============================================

This script provides quick access to all major functions of the
tennis prediction system.

Usage:
    python run.py train       # Train the model
    python run.py predict     # Run prediction app
    python run.py backtest    # Run backtesting
    python run.py features    # Generate features
"""

import sys
import os

def main():
    if len(sys.argv) < 2:
        print("🎾 Tennis Match Prediction System")
        print("================================")
        print()
        print("Usage:")
        print("  python run.py train       # Train the model")
        print("  python run.py predict     # Run prediction app")
        print("  python run.py backtest    # Run backtesting")
        print("  python run.py features    # Generate features")
        print()
        print("For web app: streamlit run match_predictor.py")
        return

    command = sys.argv[1].lower()

    if command == "train":
        print("🏋️ Training model...")
        from src.train_model import train_model
        train_model()
        
    elif command == "predict":
        print("🎮 Starting prediction app...")
        os.system("streamlit run match_predictor.py")
        
    elif command == "backtest":
        print("📈 Running backtesting...")
        from src.backtest_2025 import walk_forward_validation
        walk_forward_validation()
        
    elif command == "features":
        print("⚙️ Generating features...")
        # Import and run feature engineering if available
        try:
            from src.feature_engineering import main as generate_features
            generate_features()
        except (ImportError, AttributeError):
            print("Feature generation function not available.")
            print("Run the feature engineering script directly.")
            
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: train, predict, backtest, features")

if __name__ == "__main__":
    main()

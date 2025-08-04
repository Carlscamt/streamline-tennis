#!/usr/bin/env python3
"""
🎾 TENNIS MATCH PREDICTION - STREAMLINED LAUNCHER
================================================

Ultra-clean launcher for the tennis match prediction system.
Single command to launch the high-performance optimized app.

Usage:
    python launch.py    # Launch optimized app (300x faster)
"""

import sys
import subprocess
from pathlib import Path

def launch_optimized_app():
    """Launch the high-performance optimized Streamlit app."""
    repo_root = Path(__file__).parent.absolute()
    app_path = repo_root / "apps" / "optimized_match_predictor.py"
    
    if not app_path.exists():
        print("❌ Optimized app not found at:", app_path)
        return False
    
    print("🎾 TENNIS MATCH PREDICTION SYSTEM")
    print("🚀 Launching HIGH-PERFORMANCE Tennis Predictor...")
    print("📊 300x faster predictions | Real-time interface")
    print("🌐 Opening at: http://localhost:8501")
    print("-" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], 
                      cwd=repo_root, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching app: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
        return True

if __name__ == "__main__":
    success = launch_optimized_app()
    sys.exit(0 if success else 1)

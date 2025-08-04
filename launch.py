#!/usr/bin/env python3
"""
🎾 TENNIS MATCH PREDICTION - QUICK LAUNCHER
==========================================

Quick launcher script for the tennis match prediction system.
Handles the new organized repository structure automatically.

Usage:
    python launch.py                    # Launch optimized app
    python launch.py --legacy           # Launch original app  
    python launch.py --test             # Run comprehensive test
    python launch.py --verify           # Run performance verification
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def get_repo_root():
    """Get the repository root directory."""
    current_dir = Path(__file__).parent.absolute()
    return current_dir

def launch_optimized_app():
    """Launch the high-performance optimized Streamlit app."""
    repo_root = get_repo_root()
    app_path = repo_root / "apps" / "optimized_match_predictor.py"
    
    if not app_path.exists():
        print("❌ Optimized app not found at:", app_path)
        return False
    
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

def launch_legacy_app():
    """Legacy app has been removed - redirect to optimized app."""
    print("⚠️  Legacy app has been removed for code cleanup.")
    print("� Launching the HIGH-PERFORMANCE optimized app instead...")
    return launch_optimized_app()

def run_comprehensive_test():
    """Run the comprehensive system test."""
    repo_root = get_repo_root()
    test_path = repo_root / "tests" / "comprehensive_test.py"
    
    if not test_path.exists():
        print("❌ Comprehensive test not found at:", test_path)
        return False
    
    print("🧪 Running COMPREHENSIVE SYSTEM TEST...")
    print("📊 Testing all components and performance")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, str(test_path)], 
                              cwd=repo_root, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def run_performance_verification():
    """Run performance verification."""
    repo_root = get_repo_root()
    test_path = repo_root / "tests" / "final_verification.py"
    
    if not test_path.exists():
        print("❌ Performance verification not found at:", test_path)
        return False
    
    print("⚡ Running PERFORMANCE VERIFICATION...")
    print("📈 Benchmarking system performance")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, str(test_path)], 
                              cwd=repo_root, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running verification: {e}")
        return False

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="🎾 Tennis Match Prediction System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                    # Launch optimized app (default)
  python launch.py --legacy           # Launch original app
  python launch.py --test             # Run comprehensive test
  python launch.py --verify           # Run performance verification
  
🎯 Recommended: Use default (optimized app) for 300x faster predictions!
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--legacy", action="store_true", 
                      help="Launch original app (slower)")
    group.add_argument("--test", action="store_true",
                      help="Run comprehensive system test")  
    group.add_argument("--verify", action="store_true",
                      help="Run performance verification")
    
    args = parser.parse_args()
    
    print("🎾 TENNIS MATCH PREDICTION SYSTEM")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended for best performance")
    
    success = False
    
    if args.legacy:
        success = launch_legacy_app()
    elif args.test:
        success = run_comprehensive_test()
    elif args.verify:
        success = run_performance_verification()
    else:
        # Default: launch optimized app
        success = launch_optimized_app()
    
    if success:
        print("\n✅ Operation completed successfully!")
    else:
        print("\n❌ Operation failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()

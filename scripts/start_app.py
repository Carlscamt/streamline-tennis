"""
🎾 Tennis Prediction App Launcher
================================
Simple script to start the optimized Streamlit tennis match predictor.
"""

import subprocess
import sys
import os

def start_streamlit_app():
    """Start the Streamlit application."""
    
    print("🎾 STARTING OPTIMIZED TENNIS MATCH PREDICTOR")
    print("="*50)
    
    # Change to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Python executable path
    python_exe = os.path.join("..", ".venv", "Scripts", "python.exe")
    
    # Streamlit command
    cmd = [python_exe, "-m", "streamlit", "run", "optimized_match_predictor.py", "--server.port", "8501"]
    
    print("🚀 Launching Streamlit app...")
    print("📍 Local URL: http://localhost:8501")
    print("⚡ Features: High-performance tennis match predictions")
    print("\n💡 To stop the app: Press Ctrl+C in this terminal")
    print("="*50)
    
    try:
        # Start the Streamlit app
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
        return True
    
    return True

if __name__ == "__main__":
    start_streamlit_app()

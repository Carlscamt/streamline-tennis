"""
🎾 Minimal Tennis Predictor Test
===============================
Simple test version to verify the optimized system works.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

def main():
    st.title("🎾 Tennis Match Predictor - Test Version")
    st.write("Testing the optimized tennis prediction system...")
    
    # Test data loading
    with st.spinner("Loading tennis data..."):
        try:
            df = pd.read_csv('data/tennis_data/tennis_data.csv', low_memory=False)
            st.success(f"✅ Data loaded: {len(df):,} matches")
            
            # Show sample data
            st.subheader("📊 Sample Data")
            st.dataframe(df[['Date', 'Winner', 'Loser', 'Surface']].head(10))
            
            # Test model loading
            try:
                import joblib
                model = joblib.load('models/tennis_model.joblib')
                st.success("✅ Model loaded successfully")
            except Exception as e:
                st.error(f"❌ Model loading failed: {e}")
            
            # Test rating cache
            try:
                from rating_cache import RatingCache
                cache = RatingCache(df, use_persistence=False)
                cache_info = cache.get_cache_info()
                st.success(f"✅ Rating cache loaded: {cache_info['players_loaded']} players")
            except Exception as e:
                st.error(f"❌ Rating cache failed: {e}")
            
            # Show some statistics
            st.subheader("📈 Dataset Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                unique_players = set(df['Winner'].unique()) | set(df['Loser'].unique())
                st.metric("Unique Players", len(unique_players))
            
            with col2:
                st.metric("Surfaces", df['Surface'].nunique())
            
            with col3:
                date_range = pd.to_datetime(df['Date']).max() - pd.to_datetime(df['Date']).min()
                st.metric("Years of Data", f"{date_range.days // 365}")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sample Feature Engineering - Generate features for a sample of matches for immediate testing
"""

import pandas as pd
import sys
from datetime import datetime
sys.path.append('.')
from feature_engineering import calculate_features

def run_sample_feature_engineering(sample_size=5000):
    """Run feature engineering on a sample for quick testing"""
    print(f"🎾 SAMPLE FEATURE ENGINEERING ({sample_size:,} matches)")
    print("=" * 60)
    
    # Load raw data
    raw_data_path = '../data/tennis_data/tennis_data.csv'
    print(f"Loading raw data from: {raw_data_path}")
    
    try:
        original_df = pd.read_csv(raw_data_path, low_memory=False)
        print(f"✅ Loaded {len(original_df):,} total matches")
        
        # Take a recent sample for testing
        original_df['Date'] = pd.to_datetime(original_df['Date'])
        original_df = original_df.sort_values('Date')
        
        # Take the most recent sample_size matches
        sample_df = original_df.tail(sample_size).copy()
        print(f"📊 Processing sample: {len(sample_df):,} recent matches")
        print(f"Date range: {sample_df['Date'].min()} to {sample_df['Date'].max()}")
        
        # Run feature engineering on sample
        print("🔄 Computing features...")
        features_df = calculate_features(sample_df)
        
        print(f"✅ Generated features for {len(features_df):,} match perspectives")
        print(f"Features computed: {len([c for c in features_df.columns if c not in ['Win', 'Date', 'match_id']])}")
        
        # Save sample features
        output_path = '../data/tennis_features_sample.csv'
        features_df.to_csv(output_path, index=False)
        
        print(f"💾 Sample features saved to: {output_path}")
        
        # Show some statistics
        from features import FEATURES, ADDITIONAL_FEATURES
        ALL_FEATURES = FEATURES + ADDITIONAL_FEATURES
        
        print(f"\n📈 FEATURE STATISTICS:")
        feature_stats = []
        for feature in ALL_FEATURES[:10]:  # Show first 10
            if feature in features_df.columns:
                values = features_df[feature]
                stats = {
                    'feature': feature,
                    'min': values.min(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'std': values.std()
                }
                feature_stats.append(stats)
                print(f"{feature:<30} min={stats['min']:>8.4f} max={stats['max']:>8.4f} std={stats['std']:>8.4f}")
        
        # Check for variation
        zero_variation = [f for f in ALL_FEATURES if f in features_df.columns and features_df[f].std() == 0]
        if zero_variation:
            print(f"\n⚠️  WARNING: {len(zero_variation)} features have zero variation")
        else:
            print(f"\n✅ All features show proper variation!")
            
        return output_path
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

if __name__ == "__main__":
    result = run_sample_feature_engineering(sample_size=5000)
    if result:
        print(f"\n🎯 SUCCESS! Sample features generated at: {result}")
        print("You can now train with real feature values using:")
        print("  1. Update train_model.py to use tennis_features_sample.csv")
        print("  2. Run python train_model.py")
    else:
        print("\n❌ FAILED! Check the error messages above.")

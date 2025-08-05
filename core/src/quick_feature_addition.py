#!/usr/bin/env python3
"""
Quick Feature Addition - Add missing features with defaults for immediate testing
This allows us to test the new model architecture while full feature engineering runs
"""

import pandas as pd
from features import FEATURES, ADDITIONAL_FEATURES

def add_missing_features():
    """Add missing additional features with neutral default values"""
    print("🎾 QUICK FEATURE ADDITION FOR IMMEDIATE TESTING")
    print("=" * 60)
    
    # Load existing dataset
    df = pd.read_csv('../data/tennis_features.csv')
    print(f"Loaded dataset: {len(df)} matches with {len(df.columns)} columns")
    
    # Check which additional features are missing
    missing_features = [f for f in ADDITIONAL_FEATURES if f not in df.columns]
    existing_additional = [f for f in ADDITIONAL_FEATURES if f in df.columns]
    
    print(f"Additional features status:")
    print(f"  - Already present: {len(existing_additional)}")
    print(f"  - Missing: {len(missing_features)}")
    
    if missing_features:
        print(f"\nAdding {len(missing_features)} missing features with neutral defaults...")
        
        # Add missing features with neutral default values (0.0 for differences)
        for feature in missing_features:
            df[feature] = 0.0
            print(f"  ✅ Added: {feature}")
        
        # Save updated dataset
        output_path = '../data/tennis_features_enhanced.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ SUCCESS!")
        print(f"Enhanced dataset saved to: {output_path}")
        print(f"Total features: {len(FEATURES + ADDITIONAL_FEATURES)}")
        print(f"Dataset ready for training with all {len(ADDITIONAL_FEATURES)} additional features!")
        
        return output_path
    else:
        print("\n✅ All additional features already present!")
        return '../data/tennis_features.csv'

if __name__ == "__main__":
    output_file = add_missing_features()
    print(f"\nTo train with enhanced features, use: {output_file}")

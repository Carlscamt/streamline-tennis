import pandas as pd
from features import ADDITIONAL_FEATURES

df = pd.read_csv('../data/tennis_features_enhanced.csv')
print("FEATURE VARIATION CHECK:")
print("=" * 60)

for f in ADDITIONAL_FEATURES[:10]:
    min_val = df[f].min()
    max_val = df[f].max() 
    std_val = df[f].std()
    print(f'{f:<35} min={min_val:.4f} max={max_val:.4f} std={std_val:.4f}')

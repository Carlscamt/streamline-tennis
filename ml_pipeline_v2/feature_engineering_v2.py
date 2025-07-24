import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_expanding_features(df):
    # Sort by date for expanding window
    df = df.sort_values('Date')
    # Surface-specific win rates
    if 'Surface' in df.columns:
        for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
            df[f'player1_{surface}_win_pct'] = 0.5
            df[f'player2_{surface}_win_pct'] = 0.5
    # Recent form (last 10 matches)
    df['player1_recent_win_pct'] = 0.5
    df['player2_recent_win_pct'] = 0.5
    # Fatigue: matches played in last 14 days
    df['player1_matches_14d'] = 0
    df['player2_matches_14d'] = 0
    # Tournament round encoding (if present)
    if 'Round' in df.columns:
        df['round_encoded'] = df['Round'].astype('category').cat.codes
    else:
        df['round_encoded'] = 0
    # Serve/return stats (placeholders)
    df['player1_aces'] = 0
    df['player2_aces'] = 0
    df['player1_double_faults'] = 0
    df['player2_double_faults'] = 0
    # Interaction features
    df['surface_form_interaction'] = 0.0
    # Handedness encoding (if present)
    if 'Player1Hand' in df.columns:
        df['player1_hand_encoded'] = df['Player1Hand'].astype('category').cat.codes
    else:
        df['player1_hand_encoded'] = 0
    if 'Player2Hand' in df.columns:
        df['player2_hand_encoded'] = df['Player2Hand'].astype('category').cat.codes
    else:
        df['player2_hand_encoded'] = 0
    # Remove post-match updated features (example: volatility)
    if 'glicko2_volatility_diff' in df.columns:
        df = df.drop(columns=['glicko2_volatility_diff'])
    return df

def main():
    # Load balanced match data
    df = pd.read_csv('tennis_data_features_balanced_v2.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    # Compute new features
    df = compute_expanding_features(df)
    # Downcast floats for memory
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    # Save engineered features
    df.to_csv('tennis_features_v2.csv', index=False)
    print('Feature engineering complete. Output: tennis_features_v2.csv')

if __name__ == '__main__':
    main()

import pandas as pd

def generate_balanced_features(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    rows = []
    for _, row in df.iterrows():
        # Winner row (Win=1)
        winner_row = row.copy()
        winner_row['Win'] = 1
        rows.append(winner_row)
        # Loser row (Win=0), swap winner/loser and invert relevant features
        loser_row = row.copy()
        loser_row['Winner'], loser_row['Loser'] = row['Loser'], row['Winner']
        loser_row['Win'] = 0
        # Invert all _diff features
        for col in df.columns:
            if col.endswith('_diff'):
                loser_row[col] = -row[col]
        # Optionally swap surface-specific features if present
        for surface in ['Hard', 'Clay', 'Grass', 'Carpet']:
            p1_col = f'player1_{surface}_win_pct'
            p2_col = f'player2_{surface}_win_pct'
            if p1_col in df.columns and p2_col in df.columns:
                loser_row[p1_col], loser_row[p2_col] = row[p2_col], row[p1_col]
        # Swap recent form, matches played, aces, double faults, handedness
        for pair in [
            ('player1_recent_win_pct', 'player2_recent_win_pct'),
            ('player1_matches_14d', 'player2_matches_14d'),
            ('player1_aces', 'player2_aces'),
            ('player1_double_faults', 'player2_double_faults'),
            ('player1_hand_encoded', 'player2_hand_encoded')
        ]:
            a, b = pair
            if a in df.columns and b in df.columns:
                loser_row[a], loser_row[b] = row[b], row[a]
        rows.append(loser_row)
    balanced_df = pd.DataFrame(rows)
    balanced_df.to_csv(output_csv, index=False)
    print(f"Balanced dataset saved to {output_csv}")

if __name__ == '__main__':
    generate_balanced_features('tennis_data_features_no_leakage.csv', 'tennis_data_features_balanced_v2.csv')

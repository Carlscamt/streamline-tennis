#!/usr/bin/env python3
"""
Test script to simulate the exact workflow of making a prediction
in the Streamlit app to identify any remaining bugs.
"""

import pandas as pd
import numpy as np
import joblib
from feature_engineering import EloRating, Glicko2Rating
from betting_strategy import UncertaintyShrinkageBetting

def calculate_elo_rating(historical_data, player, surface, as_of_date):
    # Initialize with reasonable parameters for tennis
    elo = EloRating(k=32, initial_rating=1500, decay_factor=0.97)
    
    # Get all relevant matches before the given date
    relevant_matches = historical_data[
        (historical_data['Date'] < as_of_date) &
        (historical_data['Surface'] == surface)
    ].sort_values('Date')
    
    # Process each match to update Elo ratings
    for _, match in relevant_matches.iterrows():
        # Only process matches where the player participated
        if match['Winner'] == player or match['Loser'] == player:
            winner = match['Winner']
            loser = match['Loser']
            
            # Convert date string to datetime if needed
            match_date = pd.to_datetime(match['Date']) if isinstance(match['Date'], str) else match['Date']
            
            # Calculate margin of victory if sets information is available
            margin = 1.0
            elo.update_rating(winner, loser, match_date, margin_of_victory=margin)
    
    # Return the player's current Elo rating
    return elo.get_rating(player, as_of_date)

def calculate_glicko2_data(historical_data, player, surface, as_of_date):
    # Initialize Glicko-2 system
    glicko2 = Glicko2Rating(initial_rating=1500, initial_rd=350, 
                           initial_volatility=0.06, tau=0.3)
    
    # Get all relevant matches before the given date
    relevant_matches = historical_data[
        (historical_data['Date'] < as_of_date) &
        (historical_data['Surface'] == surface)
    ].sort_values('Date')
    
    # Process each match to update Glicko-2 ratings
    for _, match in relevant_matches.iterrows():
        # Only process matches where the player participated
        if match['Winner'] == player or match['Loser'] == player:
            winner = match['Winner']
            loser = match['Loser']
            
            # Convert date string to datetime if needed
            match_date = pd.to_datetime(match['Date']) if isinstance(match['Date'], str) else match['Date']
            
            # Calculate margin of victory if sets information is available
            margin = 1.0
            glicko2.update_rating(winner, loser, match_date, margin_of_victory=margin)
    
    # Return the player's current Glicko-2 data
    return glicko2.get_player_data(player, current_date=as_of_date)

def get_player_stats(historical_data, player, surface, as_of_date):
    # Get matches where player appears as either Winner or Loser
    player_matches = historical_data[
        ((historical_data['Winner'] == player) | (historical_data['Loser'] == player)) &
        (historical_data['Date'] < as_of_date)
    ].copy()
    
    # Career stats
    total_matches = len(player_matches)
    if total_matches > 0:
        wins = len(player_matches[player_matches['Winner'] == player])
        career_win_percentage = wins / total_matches
        
        # Surface stats
        surface_matches = player_matches[player_matches['Surface'] == surface]
        surface_total = len(surface_matches)
        if surface_total > 0:
            surface_wins = len(surface_matches[surface_matches['Winner'] == player])
            surface_win_percentage = surface_wins / surface_total
        else:
            surface_win_percentage = career_win_percentage  # Use career stats if no surface data
        
        # Recent form
        recent_matches = player_matches.sort_values('Date', ascending=False).head(10)
        if len(recent_matches) > 0:
            recent_wins = len(recent_matches[recent_matches['Winner'] == player])
            recent_form = recent_wins / len(recent_matches)
        else:
            recent_form = career_win_percentage  # Use career stats if no recent matches
    else:
        # Default values for new players
        career_win_percentage = 0.5
        surface_win_percentage = 0.5
        recent_form = 0.5
    
    # Calculate Elo rating
    elo_rating = calculate_elo_rating(historical_data, player, surface, as_of_date)
    
    # Calculate Glicko-2 data
    glicko2_data = calculate_glicko2_data(historical_data, player, surface, as_of_date)
    
    return career_win_percentage, surface_win_percentage, recent_form, elo_rating, glicko2_data

def get_h2h_stats(historical_data, player1, player2, as_of_date):
    h2h_matches = historical_data[
        (historical_data['Date'] < as_of_date) &
        (
            ((historical_data['Winner'] == player1) & (historical_data['Loser'] == player2)) |
            ((historical_data['Winner'] == player2) & (historical_data['Loser'] == player1))
        )
    ].copy()
    
    if len(h2h_matches) == 0:
        # Default to 50-50 if no head-to-head history
        return 0.5
    
    # Count wins for player1
    player1_wins = len(h2h_matches[h2h_matches['Winner'] == player1])
    return player1_wins / len(h2h_matches)

def test_prediction():
    print("🧪 Testing Complete Prediction Workflow")
    print("=" * 50)
    
    try:
        # Load model and data
        print("Loading model and data...")
        model = joblib.load('tennis_model.joblib')
        historical_data = pd.read_csv('tennis_features.csv')
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        print(f"✅ Model loaded: {type(model)}")
        print(f"✅ Data loaded: {historical_data.shape}")
        
        # Test with the specific players mentioned in the error
        player1 = "Nakashima B."
        player2 = "Quinn E."
        surface = "Clay"
        current_date = pd.Timestamp.now()
        
        print(f"\n🎾 Testing prediction: {player1} vs {player2} on {surface}")
        
        # Get player statistics
        print("Getting player 1 stats...")
        p1_career_win_pct, p1_surface_win_pct, p1_form, p1_elo, p1_glicko2 = get_player_stats(
            historical_data, player1, surface, current_date)
        print(f"✅ Player 1 stats: Career={p1_career_win_pct:.3f}, Surface={p1_surface_win_pct:.3f}, Form={p1_form:.3f}, ELO={p1_elo:.1f}")
        
        print("Getting player 2 stats...")
        p2_career_win_pct, p2_surface_win_pct, p2_form, p2_elo, p2_glicko2 = get_player_stats(
            historical_data, player2, surface, current_date)
        print(f"✅ Player 2 stats: Career={p2_career_win_pct:.3f}, Surface={p2_surface_win_pct:.3f}, Form={p2_form:.3f}, ELO={p2_elo:.1f}")
        
        print("Getting H2H stats...")
        h2h_win_pct = get_h2h_stats(historical_data, player1, player2, current_date)
        print(f"✅ H2H stats: {h2h_win_pct:.3f}")
        
        # Create features DataFrame
        print("Creating features...")
        features = pd.DataFrame({
            'career_win_pct_diff': [p1_career_win_pct - p2_career_win_pct],
            'surface_win_pct_diff': [p1_surface_win_pct - p2_surface_win_pct],
            'recent_form_diff': [p1_form - p2_form],
            'h2h_win_pct_diff': [h2h_win_pct - (1 - h2h_win_pct)],
            'elo_rating_diff': [p1_elo - p2_elo],
            'glicko2_rating_diff': [p1_glicko2['rating'] - p2_glicko2['rating']],
            'glicko2_rd_diff': [p2_glicko2['rd'] - p1_glicko2['rd']],
            'glicko2_volatility_diff': [p2_glicko2['volatility'] - p1_glicko2['volatility']]
        })
        print(f"✅ Features created: {features.shape}")
        print("Features:")
        for col in features.columns:
            print(f"  {col}: {features[col].iloc[0]:.4f}")
        
        # Make prediction
        print("\nMaking prediction...")
        feature_cols = ['career_win_pct_diff', 'surface_win_pct_diff', 'recent_form_diff', 
                       'h2h_win_pct_diff', 'elo_rating_diff', 'glicko2_rating_diff', 
                       'glicko2_rd_diff', 'glicko2_volatility_diff']
        win_prob = model.predict_proba(features[feature_cols])[0][1]
        print(f"✅ Prediction successful!")
        print(f"🎯 {player1} win probability: {win_prob:.1%}")
        print(f"🎯 {player2} win probability: {(1-win_prob):.1%}")
        
        # Test betting strategy
        print("\nTesting betting strategy...")
        strategy = UncertaintyShrinkageBetting()
        uncertainty = strategy.calculate_uncertainty(win_prob)
        kelly_fraction, _ = strategy.calculate_kelly_fraction(win_prob, 2.0)
        print(f"✅ Uncertainty: {uncertainty:.1%}")
        print(f"✅ Kelly fraction: {kelly_fraction:.1%}")
        
        print("\n🎉 All tests passed! The prediction system is working correctly.")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_prediction()

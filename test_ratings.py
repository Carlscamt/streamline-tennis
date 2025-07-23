#!/usr/bin/env python3
"""
Test script for Glicko-2 implementation in tennis predictor app.

This script verifies that the Glicko-2 rating system works correctly
and generates features without errors.
"""

import pandas as pd
import numpy as np
from feature_engineering import Glicko2Rating

def test_glicko2_basic():
    """Test basic Glicko-2 functionality."""
    print("Testing basic Glicko-2 functionality...")
    
    # Initialize Glicko-2 system
    glicko2 = Glicko2Rating()
    
    # Test initial player data
    player_data = glicko2.get_player_data("TestPlayer1")
    print(f"Initial player data: {player_data}")
    
    assert player_data['rating'] == 1500
    assert player_data['rd'] == 350
    assert player_data['volatility'] == 0.06
    print("✅ Initial player data test passed")
    
    # Test rating update
    match_date = pd.Timestamp('2024-01-01')
    glicko2.update_rating("Player1", "Player2", match_date, margin_of_victory=1.0)
    
    # Check that ratings were updated
    p1_data = glicko2.get_player_data("Player1", match_date)
    p2_data = glicko2.get_player_data("Player2", match_date)
    
    print(f"Player1 after win: {p1_data}")
    print(f"Player2 after loss: {p2_data}")
    
    # Winner should have higher rating than loser
    assert p1_data['rating'] > p2_data['rating']
    print("✅ Rating update test passed")

def test_glicko2_with_sample_data():
    """Test Glicko-2 with sample match data."""
    print("\nTesting Glicko-2 with sample data...")
    
    # Create sample match data
    sample_data = pd.DataFrame({
        'Date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'Winner': ['Djokovic', 'Nadal', 'Djokovic'],
        'Loser': ['Nadal', 'Federer', 'Federer'],
        'Surface': ['Hard', 'Clay', 'Hard'],
        'WRank': [1, 2, 1],
        'LRank': [2, 3, 3],
        'Wsets': [2, 2, 2],
        'Lsets': [0, 1, 0]
    })
    
    # Test feature calculation would work
    try:
        # Import our feature calculation (but don't run full calculation)
        from feature_engineering import Glicko2Rating
        
        glicko2 = Glicko2Rating()
        
        # Test with each match
        for _, match in sample_data.iterrows():
            glicko2.update_rating(
                match['Winner'], 
                match['Loser'], 
                match['Date'], 
                margin_of_victory=match['Wsets'] / (match['Wsets'] + match['Lsets'])
            )
        
        # Check final ratings
        djokovic_data = glicko2.get_player_data("Djokovic", sample_data['Date'].max())
        nadal_data = glicko2.get_player_data("Nadal", sample_data['Date'].max())
        federer_data = glicko2.get_player_data("Federer", sample_data['Date'].max())
        
        print(f"Final ratings:")
        print(f"Djokovic: {djokovic_data}")
        print(f"Nadal: {nadal_data}")
        print(f"Federer: {federer_data}")
        
        print("✅ Sample data test passed")
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        raise

def test_feature_differences():
    """Test that Glicko-2 feature differences are calculated correctly."""
    print("\nTesting Glicko-2 feature differences...")
    
    glicko2 = Glicko2Rating()
    
    # Simulate some matches
    match_date = pd.Timestamp('2024-01-01')
    
    # Player1 wins against Player2
    glicko2.update_rating("Player1", "Player2", match_date)
    
    # Get data for both players
    p1_data = glicko2.get_player_data("Player1", match_date)
    p2_data = glicko2.get_player_data("Player2", match_date)
    
    # Calculate feature differences (as would be done in the model)
    rating_diff = p1_data['rating'] - p2_data['rating']
    rd_diff = p2_data['rd'] - p1_data['rd']  # Lower RD is better
    volatility_diff = p2_data['volatility'] - p1_data['volatility']  # Lower volatility is better
    
    print(f"Rating difference (P1-P2): {rating_diff:.2f}")
    print(f"RD difference (P2-P1): {rd_diff:.2f}")
    print(f"Volatility difference (P2-P1): {volatility_diff:.4f}")
    
    # Player1 (winner) should have positive rating advantage
    assert rating_diff > 0, f"Winner should have higher rating, got {rating_diff}"
    
    print("✅ Feature differences test passed")

if __name__ == "__main__":
    print("🧪 Testing Glicko-2 Implementation")
    print("=" * 50)
    
    try:
        test_glicko2_basic()
        test_glicko2_with_sample_data()
        test_feature_differences()
        
        print("\n🎉 All Glicko-2 tests passed!")
        print("\n✅ The rating system implementation is working correctly.")
        print("\nNext steps:")
        print("1. Generate features: python feature_engineering.py")
        print("2. Train model: python train_model.py")
        print("3. Run predictions: streamlit run match_predictor.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nPlease check the implementation and try again.")

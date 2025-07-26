"""
🎾 TENNIS MATCH PREDICTION - INTERACTIVE WEB APPLICATION
======================================================

STREAMLIT-BASED TENNIS MATCH PREDICTOR
=====================================
This is the main interactive web application for predicting tennis match outcomes.
Users can select players, court surfaces, and get real-time predictions with
betting recommendations.

🚀 ULTRA-STREAMLINED ARCHITECTURE:
=================================
This app uses the ultra-streamlined feature system:
• All features computed by features.py
• No need to edit multiple files for new features
• Automatic integration with the ML model

🎯 FEATURES:
===========
• Real-time Match Predictions: Select any two players and get win probabilities
• Surface-Specific Analysis: Account for different court surfaces (Hard, Clay, Grass, Carpet)
• Rating Systems: Live ELO and Glicko-2 rating calculations
• Betting Strategy: Kelly Criterion-based betting recommendations
• Historical Context: Player statistics and head-to-head analysis
• Uncertainty Visualization: Confidence intervals and model uncertainty

📊 PREDICTION PIPELINE:
======================
1. Load trained XGBoost model and historical data
2. Calculate player statistics (win rates, ratings, recent form)
3. Compute feature vector using ultra-streamlined system
4. Generate probability predictions using trained model
5. Apply betting strategy with uncertainty shrinkage
6. Display results with confidence metrics

🎮 USER INTERFACE:
=================
Sidebar Controls:
• Player Selection: Dropdown menus with all historical players
• Surface Selection: Choose court surface type
• Date Selection: Set prediction date for historical context

Main Display:
• Match Prediction: Win probabilities for both players
• Feature Analysis: Individual feature contributions
• Betting Recommendations: Suggested bet sizes and expected returns
• Player Statistics: Recent form, ratings, head-to-head records

🔧 TECHNICAL DETAILS:
====================
Dependencies:
• Streamlit: Web interface framework
• XGBoost: Machine learning model
• Pandas/NumPy: Data processing
• Custom modules: feature_engineering, features, betting_strategy

Data Requirements:
• tennis_model.joblib: Trained XGBoost model
• tennis_features.csv: Historical match data (65,715+ matches)
• Features defined in features.py

Performance Optimizations:
• @st.cache_resource: Cached model loading
• @st.cache_data: Cached data processing
• Efficient rating calculations with date-based filtering

🚀 RUNNING THE APPLICATION:
===========================
Command: streamlit run match_predictor.py
Access: http://localhost:8501
Requirements: All dependencies in requirements.txt

💡 CUSTOMIZATION:
================
• Add new features: Edit features.py (automatically integrates)
• Modify betting strategy: Edit betting_strategy.py
• Update model: Retrain using train_model.py
• Change UI: Modify Streamlit components in this file

🔗 RELATED FILES:
================
• features.py: Feature computation system (ultra-streamlined)
• feature_engineering.py: Rating systems (ELO, Glicko-2)
• betting_strategy.py: Kelly Criterion implementation
• train_model.py: Model training pipeline
• tennis_model.joblib: Trained XGBoost model
• tennis_features.csv: Historical match database
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from src.feature_engineering import EloRating, Glicko2Rating
from src.features import FEATURES, build_prediction_feature_vector
from src.betting_strategy import UncertaintyShrinkageBetting

# Load the trained model and preprocessed data
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('models/tennis_model.joblib')  # Load from models directory
        historical_data = pd.read_csv('data/tennis_features.csv')  # Load from data directory
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        return model, historical_data
    except FileNotFoundError:
        st.error("Model or data files not found. Running model training first...")
        import src.train_model as train_model
        train_model.train_model()
        st.success("Model trained successfully. Please refresh the page.")
        st.stop()

# Get unique players and surfaces
@st.cache_data
def get_unique_values(historical_data):
    players = sorted(list(set(historical_data['Winner'].unique()) | set(historical_data['Loser'].unique())))
    surfaces = sorted(historical_data['Surface'].unique())
    return players, surfaces

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
            if 'Wsets' in match.index and 'Lsets' in match.index:
                try:
                    margin = float(match['Wsets']) / (float(match['Wsets']) + float(match['Lsets']))
                except (ValueError, ZeroDivisionError):
                    margin = 1.0
            else:
                margin = 1.0
                
            elo.update_rating(winner, loser, match_date, margin_of_victory=margin)
    
    # Return the player's current Elo rating with decay
    return elo.get_rating(player, current_date=as_of_date)

def calculate_glicko2_data(historical_data, player, surface, as_of_date):
    """
    Calculate player's Glicko-2 rating, RD, and volatility for a specific surface.
    
    Args:
        historical_data: DataFrame with historical match data
        player: Player name
        surface: Court surface (Hard, Clay, Grass)
        as_of_date: Date to calculate rating as of
        
    Returns:
        Dict with rating, rd, and volatility
    """
    # Initialize Glicko-2 system with reasonable parameters for tennis
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
            margin = 1.0  # Default
            if 'Wsets' in match.index and 'Lsets' in match.index:
                try:
                    w_sets = float(match['Wsets'])
                    l_sets = float(match['Lsets'])
                    if w_sets + l_sets > 0:
                        margin = w_sets / (w_sets + l_sets)
                except (ValueError, ZeroDivisionError):
                    margin = 1.0
                    
            glicko2.update_rating(winner, loser, match_date, margin_of_victory=margin)
    
    # Return the player's current Glicko-2 data
    return glicko2.get_player_data(player, current_date=as_of_date)

def get_comprehensive_player_stats(historical_data, player, surface, as_of_date):
    """Get comprehensive player statistics needed for all features."""
    # Get matches where player appears as either Winner or Loser
    player_matches = historical_data[
        ((historical_data['Winner'] == player) | (historical_data['Loser'] == player)) &
        (historical_data['Date'] < as_of_date)
    ].copy().sort_values('Date')
    
    # Initialize defaults
    career_win_percentage = 0.5
    surface_win_percentage = 0.5
    recent_form = 0.5
    win_streak = 0
    loss_streak = 0
    fatigue_days = 99  # Default high fatigue if no matches
    surface_adaptability = 0.0
    h2h_surface_win_pct = 0.5
    
    total_matches = len(player_matches)
    if total_matches > 0:
        # Career stats
        wins = len(player_matches[player_matches['Winner'] == player])
        career_win_percentage = wins / total_matches
        
        # Surface stats
        surface_matches = player_matches[player_matches['Surface'] == surface]
        surface_total = len(surface_matches)
        if surface_total > 0:
            surface_wins = len(surface_matches[surface_matches['Winner'] == player])
            surface_win_percentage = surface_wins / surface_total
        else:
            surface_win_percentage = career_win_percentage
        
        # Recent form (last 10 matches)
        recent_matches = player_matches.sort_values('Date', ascending=False).head(10)
        if len(recent_matches) > 0:
            recent_wins = len(recent_matches[recent_matches['Winner'] == player])
            recent_form = recent_wins / len(recent_matches)
        else:
            recent_form = career_win_percentage
        
        # Calculate streaks - look at most recent matches
        recent_sorted = player_matches.sort_values('Date', ascending=False)
        current_win_streak = 0
        current_loss_streak = 0
        
        for _, match in recent_sorted.iterrows():
            if match['Winner'] == player:
                if current_loss_streak == 0:  # Still in win streak
                    current_win_streak += 1
                else:
                    break
            else:
                if current_win_streak == 0:  # Still in loss streak
                    current_loss_streak += 1
                else:
                    break
        
        win_streak = current_win_streak
        loss_streak = current_loss_streak
        
        # Fatigue (days since last match)
        if len(player_matches) > 0:
            last_match_date = player_matches['Date'].max()
            fatigue_days = (as_of_date - last_match_date).days
        
        # Surface adaptability (variance across surfaces)
        surface_stats = {}
        for surf in historical_data['Surface'].unique():
            surf_matches = player_matches[player_matches['Surface'] == surf]
            if len(surf_matches) > 0:
                surf_wins = len(surf_matches[surf_matches['Winner'] == player])
                surface_stats[surf] = surf_wins / len(surf_matches)
        
        if len(surface_stats) > 1:
            surface_adaptability = np.var(list(surface_stats.values()))
    
    # Calculate Elo rating
    elo_rating = calculate_elo_rating(historical_data, player, surface, as_of_date)
    
    # Calculate Glicko-2 data
    glicko2_data = calculate_glicko2_data(historical_data, player, surface, as_of_date)
    
    return {
        'career_win_pct': career_win_percentage,
        'surface_win_pct': surface_win_percentage,
        'recent_form': recent_form,
        'elo_rating': elo_rating,
        'glicko2_rating': glicko2_data['rating'],
        'glicko2_rd': glicko2_data['rd'],
        'glicko2_volatility': glicko2_data['volatility'],
        'win_streak': win_streak,
        'loss_streak': loss_streak,
        'fatigue_days': fatigue_days,
        'surface_adaptability': surface_adaptability,
        'h2h_surface_win_pct': h2h_surface_win_pct  # Will be updated with H2H function
    }

def get_comprehensive_h2h_stats(historical_data, player1, player2, surface, as_of_date):
    """Get comprehensive head-to-head statistics."""
    h2h_matches = historical_data[
        (historical_data['Date'] < as_of_date) &
        (
            ((historical_data['Winner'] == player1) & (historical_data['Loser'] == player2)) |
            ((historical_data['Winner'] == player2) & (historical_data['Loser'] == player1))
        )
    ].copy()
    
    # Overall H2H
    if len(h2h_matches) == 0:
        player1_h2h_win_pct = 0.5
        player2_h2h_win_pct = 0.5
    else:
        player1_wins = len(h2h_matches[h2h_matches['Winner'] == player1])
        player1_h2h_win_pct = player1_wins / len(h2h_matches)
        player2_h2h_win_pct = 1 - player1_h2h_win_pct
    
    # Surface-specific H2H
    surface_h2h_matches = h2h_matches[h2h_matches['Surface'] == surface]
    if len(surface_h2h_matches) == 0:
        player1_h2h_surface_win_pct = 0.5
        player2_h2h_surface_win_pct = 0.5
    else:
        surface_player1_wins = len(surface_h2h_matches[surface_h2h_matches['Winner'] == player1])
        player1_h2h_surface_win_pct = surface_player1_wins / len(surface_h2h_matches)
        player2_h2h_surface_win_pct = 1 - player1_h2h_surface_win_pct
    
    return {
        'player1_h2h_win_pct': player1_h2h_win_pct,
        'player2_h2h_win_pct': player2_h2h_win_pct,
        'player1_h2h_surface_win_pct': player1_h2h_surface_win_pct,
        'player2_h2h_surface_win_pct': player2_h2h_surface_win_pct
    }

def get_player_stats(historical_data, player, surface, as_of_date):
    """Legacy function for backward compatibility."""
    stats = get_comprehensive_player_stats(historical_data, player, surface, as_of_date)
    return (stats['career_win_pct'], stats['surface_win_pct'], stats['recent_form'], 
            stats['elo_rating'], {'rating': stats['glicko2_rating'], 'rd': stats['glicko2_rd'], 
                                 'volatility': stats['glicko2_volatility']})

def get_h2h_stats(historical_data, player1, player2, as_of_date):
    """Legacy function for backward compatibility."""
    # Just get overall H2H for any surface
    h2h_data = get_comprehensive_h2h_stats(historical_data, player1, player2, 'Hard', as_of_date)
    return h2h_data['player1_h2h_win_pct']

def main():
    st.title("Tennis Match Predictor")

    # Initialize session state variables
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'market_odds' not in st.session_state:
        st.session_state.market_odds = 2.0
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 1000
    if 'risk_factor' not in st.session_state:
        st.session_state.risk_factor = 0.25

    try:
        model, historical_data = load_model_and_data()
        if model is None:
            return

        players, surfaces = get_unique_values(historical_data)

        strategy = UncertaintyShrinkageBetting(
            initial_bankroll=st.session_state.bankroll,
            min_prob=0.55,
            shrinkage_factor=0.25,
            uncertainty_threshold=0.4
        )

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Player 1")
                search_player1 = st.text_input("Search Player 1", key="search_player1").lower()
                filtered_players1 = [p for p in players if search_player1 in p.lower()] if search_player1 else players
                player1 = st.selectbox("Select Player 1", filtered_players1, key="player1")
            with col2:
                st.subheader("Player 2")
                search_player2 = st.text_input("Search Player 2", key="search_player2").lower()
                filtered_players2 = [p for p in players if search_player2 in p.lower()] if search_player2 else players
                player2 = st.selectbox("Select Player 2", filtered_players2, key="player2")
            
            surface = st.selectbox("Select Surface", surfaces, key="surface")
            submit = st.form_submit_button("Predict Match")

        if submit:
            if player1 and player2 and surface:
                if player1 == player2:
                    st.error("Please select different players")
                else:
                    current_date = pd.Timestamp.now()
                    
                    # Get comprehensive stats for both players
                    p1_stats = get_comprehensive_player_stats(historical_data, player1, surface, current_date)
                    p2_stats = get_comprehensive_player_stats(historical_data, player2, surface, current_date)
                    
                    # Get H2H stats
                    h2h_stats = get_comprehensive_h2h_stats(historical_data, player1, player2, surface, current_date)
                    
                    # Update H2H specific stats
                    p1_stats['h2h_win_pct'] = h2h_stats['player1_h2h_win_pct']
                    p2_stats['h2h_win_pct'] = h2h_stats['player2_h2h_win_pct']
                    p1_stats['h2h_surface_win_pct'] = h2h_stats['player1_h2h_surface_win_pct']
                    p2_stats['h2h_surface_win_pct'] = h2h_stats['player2_h2h_surface_win_pct']

                    # Build feature vector using centralized computation
                    feature_values = build_prediction_feature_vector(p1_stats, p2_stats)
                    features = pd.DataFrame([feature_values])

                    feature_cols = FEATURES
                    win_prob = model.predict_proba(features[feature_cols])[0][1]

                    # Store results in session state with unique keys
                    st.session_state.prediction_made = True
                    st.session_state.predicted_win_prob = win_prob
                    st.session_state.predicted_player1 = player1
                    st.session_state.predicted_player2 = player2
                    st.session_state.predicted_surface = surface
                    st.session_state.predicted_p1_stats = p1_stats
                    st.session_state.predicted_p2_stats = p2_stats
                    st.session_state.predicted_h2h_stats = h2h_stats

        if st.session_state.prediction_made:
            # Retrieve results from session state
            win_prob = st.session_state.predicted_win_prob
            player1 = st.session_state.predicted_player1
            player2 = st.session_state.predicted_player2
            surface = st.session_state.predicted_surface
            p1_stats = st.session_state.predicted_p1_stats
            p2_stats = st.session_state.predicted_p2_stats
            h2h_stats = st.session_state.predicted_h2h_stats

            # Extract individual stats for display (backward compatibility)
            p1_elo = p1_stats['elo_rating']
            p2_elo = p2_stats['elo_rating']
            p1_glicko2 = {'rating': p1_stats['glicko2_rating'], 'rd': p1_stats['glicko2_rd'], 'volatility': p1_stats['glicko2_volatility']}
            p2_glicko2 = {'rating': p2_stats['glicko2_rating'], 'rd': p2_stats['glicko2_rd'], 'volatility': p2_stats['glicko2_volatility']}
            p1_career_win_pct = p1_stats['career_win_pct']
            p2_career_win_pct = p2_stats['career_win_pct']
            p1_surface_win_pct = p1_stats['surface_win_pct']
            p2_surface_win_pct = p2_stats['surface_win_pct']
            p1_form = p1_stats['recent_form']
            p2_form = p2_stats['recent_form']
            h2h_win_pct = h2h_stats['player1_h2h_win_pct']

            # Determine the probability of the predicted winner and the predicted winner's name
            if win_prob >= 0.5:
                predicted_winner_prob = win_prob
                predicted_winner_name = player1
                predicted_loser_name = player2
            else:
                predicted_winner_prob = 1 - win_prob
                predicted_winner_name = player2
                predicted_loser_name = player1

            st.subheader("Match Prediction")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{player1} Win Probability", f"{win_prob:.1%}")
            with col2:
                st.metric(f"{player2} Win Probability", f"{(1-win_prob):.1%}")
            
            st.subheader("Betting Analysis with Uncertainty Management")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Market Odds")
                st.session_state.market_odds = st.number_input(
                    f"Odds for {predicted_winner_name}:",
                    min_value=1.01,
                    max_value=10.0,
                    value=st.session_state.market_odds,
                    step=0.01,
                    help="Enter the current market odds for the predicted winner",
                    key='market_odds_input'
                )

            with col2:
                st.session_state.bankroll = st.number_input(
                    "Current bankroll ($):",
                    min_value=10,
                    value=st.session_state.bankroll,
                    step=10
                )

            with col3:
                st.session_state.risk_factor = st.slider(
                    "Risk Factor:",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.risk_factor,
                    step=0.05,
                    help="Adjusts the aggressiveness of the betting strategy. Lower values are more conservative."
                )

            uncertainty = strategy.calculate_uncertainty(predicted_winner_prob)
            
            if predicted_winner_prob > strategy.min_prob:
                kelly_fraction, _ = strategy.calculate_kelly_fraction(predicted_winner_prob, st.session_state.market_odds)
                recommended_bet = kelly_fraction * st.session_state.bankroll * st.session_state.risk_factor
                
                confidence_color = "red" if uncertainty > 0.7 else "orange" if uncertainty > 0.5 else "green"
                confidence_level = "Low" if uncertainty > 0.7 else "Medium" if uncertainty > 0.5 else "High"
                
                st.markdown(f"""
                ### Match Analysis
                - **Win Probability**: {predicted_winner_prob:.2%}
                - **Uncertainty Level**: {uncertainty:.2%}
                - **Confidence Level**: <span style='color: {confidence_color}'>{confidence_level}</span>
                
                ### Betting Recommendation
                - **Recommended Bet**: ${recommended_bet:.2f} ({kelly_fraction:.2%} of bankroll)
                - **Market Odds**: {st.session_state.market_odds:.2f}
                - **Expected Value (Raw)**: {(predicted_winner_prob * st.session_state.market_odds - 1):.2%}
                - **Expected Value (Adjusted)**: {((1-uncertainty) * predicted_winner_prob * st.session_state.market_odds - 1):.2%}
                """, unsafe_allow_html=True)
                
                if predicted_winner_prob * st.session_state.market_odds > 1.1:
                    if uncertainty < 0.4:
                        st.success("✅ Strong betting opportunity detected. Good odds with low uncertainty.")
                    elif uncertainty < 0.6:
                        st.warning("⚠️ Moderate betting opportunity. Consider reducing stake due to uncertainty.")
                    else:
                        st.error("🚫 High uncertainty detected. Betting not recommended despite favorable odds.")
                else:
                    st.error("🚫 Insufficient value at current odds. No bet recommended.")
            else:
                st.error("🚫 Win probability too low or uncertainty too high. No bet recommended.")

            st.subheader("Player Statistics")
            
            elo_diff = p1_elo - p2_elo
            elo_advantage = player1 if elo_diff > 0 else player2
            elo_exp_win = 1 / (1 + 10 ** (-abs(elo_diff) / 400))
            
            # Glicko-2 comparison
            glicko2_rating_diff = p1_glicko2['rating'] - p2_glicko2['rating']
            glicko2_advantage = player1 if glicko2_rating_diff > 0 else player2
            
            # Calculate Glicko-2 win probability (similar to ELO)
            glicko2_exp_win = 1 / (1 + 10 ** (-abs(glicko2_rating_diff) / 400))
            
            stats_df = pd.DataFrame({
                'Statistic': [
                    'ELO Rating (Surface)', 
                    'Glicko-2 Rating', 
                    'Glicko-2 RD (Uncertainty)', 
                    'Glicko-2 Volatility', 
                    'Career Win Rate', 
                    f'{surface} Court Win Rate', 
                    'Recent Form (Last 10)', 
                    'Head-to-Head Win Rate'
                ],
                player1: [
                    f'{p1_elo:.0f} ({"+" if elo_diff > 0 else ""}{elo_diff:.0f})', 
                    f'{p1_glicko2["rating"]:.0f} ({"+" if glicko2_rating_diff > 0 else ""}{glicko2_rating_diff:.0f})', 
                    f'{p1_glicko2["rd"]:.0f}', 
                    f'{p1_glicko2["volatility"]:.3f}', 
                    f'{p1_career_win_pct:.1%}', 
                    f'{p1_surface_win_pct:.1%}', 
                    f'{p1_form:.1%}', 
                    f'{h2h_win_pct:.1%}'
                ],
                player2: [
                    f'{p2_elo:.0f}', 
                    f'{p2_glicko2["rating"]:.0f}', 
                    f'{p2_glicko2["rd"]:.0f}', 
                    f'{p2_glicko2["volatility"]:.3f}', 
                    f'{p2_career_win_pct:.1%}', 
                    f'{p2_surface_win_pct:.1%}', 
                    f'{p2_form:.1%}', 
                    f'{(1-h2h_win_pct):.1%}'
                ]
            })
            
            st.table(stats_df)
            
            # Rating system insights
            col1, col2 = st.columns(2)
            with col1:
                if abs(elo_diff) > 100:
                    st.info(f"📊 **ELO Analysis**: {elo_advantage} would have a {elo_exp_win:.1%} chance of winning based on ELO ratings alone")
            
            with col2:
                if abs(glicko2_rating_diff) > 100:
                    st.info(f"🎯 **Glicko-2 Analysis**: {glicko2_advantage} would have a {glicko2_exp_win:.1%} chance of winning based on Glicko-2 ratings")
            
            # Rating uncertainty analysis
            avg_rd = (p1_glicko2['rd'] + p2_glicko2['rd']) / 2
            if avg_rd > 200:
                st.warning(f"⚠️ **High Rating Uncertainty**: Average RD is {avg_rd:.0f}. Results may be less reliable due to limited recent match data.")
            elif avg_rd < 100:
                st.success(f"✅ **High Rating Confidence**: Average RD is {avg_rd:.0f}. Both players have established ratings with low uncertainty.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
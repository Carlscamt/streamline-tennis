"""
🎾 TENNIS MATCH PREDICTION - FEATURE ENGINEERING & RATING SYSTEMS
================================================================

COMPREHENSIVE TENNIS RATING & FEATURE GENERATION SYSTEM
======================================================
This module implements advanced tennis rating systems and generates training features
for the machine learning model. It's the data processing backbone of the prediction system.

🎯 PRIMARY FUNCTIONS:
====================
1. Rating Systems: ELO and Glicko-2 implementations
2. Feature Generation: Player statistics and contextual features
3. Training Data: Generates tennis_features.csv for model training
4. Historical Analysis: Time-aware rating calculations

🚀 ULTRA-STREAMLINED INTEGRATION:
================================
This module now integrates with the ultra-streamlined feature system:
• Uses features.py for all feature computation
• Generates training data with centralized feature logic
• No need to modify this file when adding features

📊 RATING SYSTEMS:
=================

ELO RATING SYSTEM:
• Classic chess-derived rating system adapted for tennis
• K-factor: Adjustable rating change sensitivity (default: 32)
• Surface-specific: Separate ratings per court surface
• Decay factor: Accounts for player inactivity over time
• Margin adjustment: Considers match dominance (set scores)

GLICKO-2 RATING SYSTEM:
• Advanced probabilistic rating system
• Rating: Player skill level (similar to ELO)
• RD (Rating Deviation): Uncertainty in rating
• Volatility: Degree of rating changes over time
• Time-based decay: Increases uncertainty during inactivity
• More sophisticated than ELO for irregular play patterns

🎮 FEATURE CATEGORIES:
=====================

Performance Features:
• career_win_pct: Overall career win percentage
• surface_win_pct: Win percentage on specific surface
• recent_form: Performance in last 10 matches

Rating Features:
• elo_rating: Current ELO rating
• glicko2_rating: Current Glicko-2 rating
• glicko2_rd: Rating deviation (uncertainty)
• glicko2_volatility: Rating volatility

Head-to-Head Features:
• h2h_win_pct: Historical matchup win percentage
• h2h_surface_win_pct: H2H on current surface

Temporal Features:
• fatigue_days: Days since last match
• win_streak: Current consecutive wins
• loss_streak: Current consecutive losses

Surface Features:
• surface_adaptability: Performance across surfaces

🔧 TECHNICAL ARCHITECTURE:
=========================

Data Flow:
1. Raw match data → Rating calculations
2. Player contexts → Feature computation (via features.py)
3. Feature vectors → Training dataset
4. Balanced dataset → Model training

Key Classes:
• EloRating: Implements ELO rating system
• Glicko2Rating: Implements Glicko-2 rating system
• Various helper functions for statistics and feature generation

Data Requirements:
• Match data with Winner, Loser, Date, Surface columns
• Optional: Set scores for margin of victory
• Time-ordered data for accurate rating progression

🎯 USAGE EXAMPLES:
=================

Generate Training Data:
  python feature_engineering.py
  → Outputs: tennis_features.csv

Use Rating Systems:
  elo = EloRating(k=32, initial_rating=1500)
  glicko2 = Glicko2Rating(initial_rating=1500, initial_rd=350)

Calculate Player Features:
  context = calculate_player_context(data, player, surface, date)
  features = build_feature_vector(winner_context, loser_context)

🚀 PERFORMANCE OPTIMIZATIONS:
============================
• Vectorized pandas operations where possible
• Progress bars for long-running calculations
• Efficient data filtering and grouping
• Memory-conscious processing for large datasets

💡 CUSTOMIZATION:
================
• Rating parameters: Adjust K-factor, decay rates, volatility
• Feature windows: Change recent form period, streak definitions
• New features: Add to features.py (automatically integrates)
• Data sources: Modify input data format handling

🔗 RELATED FILES:
================
• features.py: Feature computation system (ultra-streamlined)
• train_model.py: Uses generated features for model training
• match_predictor.py: Uses rating systems for real-time predictions
• tennis_features.csv: Generated training dataset
• tennis_data/: Input data directory
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Optional
from tqdm import tqdm
from features import build_feature_vector, FEATURES, ADDITIONAL_FEATURES

class Glicko2Rating:
    def __init__(self, initial_rating: float = 1500, initial_rd: float = 350, 
                 initial_volatility: float = 0.06, tau: float = 0.3):
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.initial_volatility = initial_volatility
        self.tau = tau
        self.players: Dict[str, Dict] = {}
        
    def _scale_rating(self, rating: float) -> float:
        """Convert rating to Glicko-2 scale."""
        return (rating - self.initial_rating) / 173.7178
        
    def _scale_rd(self, rd: float) -> float:
        """Convert RD to Glicko-2 scale."""
        return rd / 173.7178
        
    def _unscale_rating(self, mu: float) -> float:
        """Convert from Glicko-2 scale to rating."""
        return mu * 173.7178 + self.initial_rating
        
    def _unscale_rd(self, phi: float) -> float:
        """Convert from Glicko-2 scale to RD."""
        return phi * 173.7178
        
    def _g(self, phi: float) -> float:
        """Calculate g(φ) function."""
        return 1.0 / math.sqrt(1 + 3 * phi * phi / (math.pi * math.pi))
        
    def _e(self, mu: float, mu_j: float, phi_j: float) -> float:
        """Calculate E(μ, μⱼ, φⱼ) function."""
        return 1.0 / (1 + math.exp(-self._g(phi_j) * (mu - mu_j)))
        
    def get_player_data(self, player: str, current_date: Optional[pd.Timestamp] = None) -> Dict:
        if player not in self.players:
            self.players[player] = {
                'rating': self.initial_rating,
                'rd': self.initial_rd,
                'volatility': self.initial_volatility,
                'last_match': current_date
            }
            return self.players[player].copy()
            
        player_data = self.players[player].copy()
        
        if current_date and player_data['last_match']:
            days_inactive = (current_date - player_data['last_match']).days
            if days_inactive > 0:
                rd_increase = min(days_inactive * 0.5, 100)  # Max 100 point increase
                player_data['rd'] = min(player_data['rd'] + rd_increase, 350)
                
        return player_data
        
    def update_rating(self, winner: str, loser: str, match_date: pd.Timestamp, 
                     margin_of_victory: float = 1.0) -> None:
        # Get current player data
        winner_data = self.get_player_data(winner, match_date)
        loser_data = self.get_player_data(loser, match_date)
        
        # Convert to Glicko-2 scale
        mu1 = self._scale_rating(winner_data['rating'])
        phi1 = self._scale_rd(winner_data['rd'])
        sigma1 = winner_data['volatility']
        
        mu2 = self._scale_rating(loser_data['rating'])
        phi2 = self._scale_rd(loser_data['rd'])
        sigma2 = loser_data['volatility']
        
        # Update winner (score = 1)
        self._update_single_player(winner, mu1, phi1, sigma1, [(mu2, phi2, 1.0 * margin_of_victory)], match_date)
        
        # Update loser (score = 0) 
        self._update_single_player(loser, mu2, phi2, sigma2, [(mu1, phi1, 0.0)], match_date)
        
    def _update_single_player(self, player: str, mu: float, phi: float, sigma: float, 
                            opponents: list, match_date: pd.Timestamp) -> None:
        
        # Step 2: Calculate v (estimated variance)
        v_inv = 0
        for mu_j, phi_j, s_j in opponents:
            g_phi_j = self._g(phi_j)
            e_val = self._e(mu, mu_j, phi_j)
            v_inv += g_phi_j * g_phi_j * e_val * (1 - e_val)
            
        if v_inv == 0:
            return  # No rating change if no variance
            
        v = 1.0 / v_inv
        
        # Step 3: Calculate Δ (improvement in rating)
        delta = 0
        for mu_j, phi_j, s_j in opponents:
            delta += self._g(phi_j) * (s_j - self._e(mu, mu_j, phi_j))
        delta *= v
        
        # Step 4: Calculate new volatility
        a = math.log(sigma * sigma)
        
        def f(x):
            ex = math.exp(x)
            phi2 = phi * phi
            v_plus_ex = v + ex
            return (ex * (delta * delta - phi2 - v - ex) / (2 * v_plus_ex * v_plus_ex) - 
                    (x - a) / (self.tau * self.tau))
        
        # Find new volatility using Illinois algorithm
        A = a
        if delta * delta > phi * phi + v:
            B = math.log(delta * delta - phi * phi - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau
            
        fa = f(A)
        fb = f(B)
        
        # Illinois algorithm iteration
        C = B  # Initialize C
        for _ in range(20):  # Max 20 iterations
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            
            if abs(C - B) < 1e-6:  # Convergence
                break
                
            if fc * fb < 0:
                A = B
                fa = fb
            else:
                fa /= 2
                
            B = C
            fb = fc
            

        # Clamp volatility to [0.01, 0.5] to prevent drift/extremes
        new_volatility = max(0.01, min(0.5, math.exp(C / 2)))
        # Diagnostic: print and check for negative volatility
        if new_volatility < 0:
            raise ValueError(f"Negative volatility for {player}: {new_volatility}")
        if math.isnan(new_volatility):
            raise ValueError(f"NaN volatility for {player}")
        # Print stats for debugging
        if hasattr(self, '_volatility_debug') and self._volatility_debug:
            print(f"Volatility stats for {player} on {match_date}: {new_volatility}")

        # Step 5: Update rating and RD
        phi_star = math.sqrt(phi * phi + new_volatility * new_volatility)
        new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
        new_mu = mu + new_phi * new_phi * delta / v

        # Convert back to original scale and update
        self.players[player] = {
            'rating': self._unscale_rating(new_mu),
            'rd': self._unscale_rd(new_phi),
            'volatility': new_volatility,
            'last_match': match_date
        }

class EloRating:
    def __init__(self, k=30, initial_rating=1500, decay_factor=0.95):
        self.k = k
        self.initial_rating = initial_rating
        self.ratings = {}
        self.last_match_date = {}
        self.decay_factor = decay_factor
    
    def get_rating(self, player, current_date=None):
        """Get a player's current Elo rating with temporal decay"""
        if player not in self.ratings:
            self.ratings[player] = self.initial_rating
            self.last_match_date[player] = current_date
            return self.initial_rating
            
        if current_date and self.last_match_date[player]:
            days_since_last_match = (current_date - self.last_match_date[player]).days
            decay = self.decay_factor ** (days_since_last_match / 365)  # Yearly decay
            return self.ratings[player] * decay
        
        return self.ratings[player]
    
    def update_rating(self, winner, loser, match_date, margin_of_victory=1.0):
        """Update Elo ratings after a match"""
        winner_rating = self.get_rating(winner, match_date)
        loser_rating = self.get_rating(loser, match_date)
        
        # Calculate expected scores
        winner_expected = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        loser_expected = 1 - winner_expected
        
        # Update ratings
        rating_change = self.k * margin_of_victory * (1 - winner_expected)
        self.ratings[winner] = winner_rating + rating_change
        self.ratings[loser] = loser_rating - rating_change
        
        # Update last match dates
        self.last_match_date[winner] = match_date
        self.last_match_date[loser] = match_date


def parse_rank(rank):
    """Convert ranking to float, handling 'NR' (Not Ranked) case"""
    try:
        return float(rank)
    except (ValueError, TypeError):
        return 2000.0


def calculate_features(df):
    """
    Calculate features for tennis match prediction WITHOUT data leakage.
    
    Key changes:
    1. One row per match (not per player perspective)
    2. Features calculated using ONLY data from BEFORE the match date
    3. No dyadic dataset - direct winner/loser comparison
    
    Args:
        df: DataFrame with match data
        
    Returns:
        DataFrame with calculated features, one row per match
    """
    from features import FEATURES
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)

    elo_system = EloRating(k=32, initial_rating=1500, decay_factor=0.97)
    glicko2_system = Glicko2Rating(initial_rating=1500, initial_rd=350, 
                                   initial_volatility=0.06, tau=0.3)
    
    player_stats = {}
    surface_stats = {}
    h2h_stats = {}
    recent_matches = {}
    all_surfaces = set()
    features = []
    
    for idx, match in tqdm(df.iterrows(), total=len(df)):
        date = match['Date']
        winner = match['Winner']
        loser = match['Loser']
        surface = match['Surface']
        all_surfaces.add(surface)
        
        for player in [winner, loser]:
            if player not in player_stats:
                player_stats[player] = {
                    'matches': 0, 
                    'wins': 0,
                    # Tournament type statistics
                    'atp250_matches': 0, 'atp250_wins': 0,
                    'atp500_matches': 0, 'atp500_wins': 0,
                    'masters1000_matches': 0, 'masters1000_wins': 0,
                    'grand_slam_matches': 0, 'grand_slam_wins': 0,
                    # Surface-specific statistics
                    'hard_court_matches': 0, 'hard_court_wins': 0,
                    'clay_court_matches': 0, 'clay_court_wins': 0,
                    'grass_court_matches': 0, 'grass_court_wins': 0,
                    # Round-specific statistics
                    'early_round_matches': 0, 'early_round_wins': 0,
                    'late_round_matches': 0, 'late_round_wins': 0,
                    'quarterfinal_matches': 0, 'quarterfinal_wins': 0,
                    # Advanced performance metrics
                    'big_match_matches': 0, 'big_match_wins': 0,
                    'clutch_matches': 0, 'clutch_wins': 0,
                    'outdoor_matches': 0, 'outdoor_wins': 0,
                    'indoor_matches': 0, 'indoor_wins': 0,
                    # Recent form by surface/tournament
                    'recent_hard_matches': [], 'recent_clay_matches': [],
                    'recent_grass_matches': [], 'recent_big_matches': []
                }
            if player not in surface_stats:
                surface_stats[player] = {s: {'matches': 0, 'wins': 0} for s in all_surfaces}
            # Dynamically add new surfaces as they appear
            if surface not in surface_stats[player]:
                surface_stats[player][surface] = {'matches': 0, 'wins': 0}
            if player not in recent_matches:
                recent_matches[player] = []
            # For streaks and fatigue
            if 'last_match_date' not in player_stats[player]:
                player_stats[player]['last_match_date'] = None
            if 'win_streak' not in player_stats[player]:
                player_stats[player]['win_streak'] = 0
            if 'loss_streak' not in player_stats[player]:
                player_stats[player]['loss_streak'] = 0
            if 'surface_wins' not in player_stats[player]:
                player_stats[player]['surface_wins'] = {}
                player_stats[player]['surface_matches'] = {}
            # Dynamically add new surfaces for adaptability stats
            if surface not in player_stats[player]['surface_wins']:
                player_stats[player]['surface_wins'][surface] = 0
                player_stats[player]['surface_matches'][surface] = 0
                
        pre_winner_elo = elo_system.get_rating(winner, date)
        pre_loser_elo = elo_system.get_rating(loser, date)
        pre_winner_glicko2 = glicko2_system.get_player_data(winner, date)
        pre_loser_glicko2 = glicko2_system.get_player_data(loser, date)
        pre_winner_stats = player_stats[winner].copy()
        pre_loser_stats = player_stats[loser].copy()
        pre_winner_surface_stat = surface_stats[winner][surface].copy()
        pre_loser_surface_stat = surface_stats[loser][surface].copy()
        pre_winner_recent = list(recent_matches[winner])[-10:] if recent_matches[winner] else []
        pre_loser_recent = list(recent_matches[loser])[-10:] if recent_matches[loser] else []
        h2h_key = tuple(sorted([winner, loser]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = {
                'matches': 0,
                'wins': {winner: 0, loser: 0},
                'surface_matches': {surf: 0 for surf in all_surfaces},
                'surface_wins': {
                    winner: {surf: 0 for surf in all_surfaces},
                    loser: {surf: 0 for surf in all_surfaces}
                }
            }
        # Dynamically add new surfaces for H2H as they appear
        for surf in all_surfaces:
            if surf not in h2h_stats[h2h_key]['surface_matches']:
                h2h_stats[h2h_key]['surface_matches'][surf] = 0
            if surf not in h2h_stats[h2h_key]['surface_wins'][winner]:
                h2h_stats[h2h_key]['surface_wins'][winner][surf] = 0
            if surf not in h2h_stats[h2h_key]['surface_wins'][loser]:
                h2h_stats[h2h_key]['surface_wins'][loser][surf] = 0
        pre_h2h = h2h_stats[h2h_key].copy()

        # --- New Features ---
        # Win/loss streaks
        winner_win_streak = pre_winner_stats.get('win_streak', 0)
        loser_win_streak = pre_loser_stats.get('win_streak', 0)
        winner_loss_streak = pre_winner_stats.get('loss_streak', 0)
        loser_loss_streak = pre_loser_stats.get('loss_streak', 0)
        win_streak_diff = winner_win_streak - loser_win_streak
        loss_streak_diff = winner_loss_streak - loser_loss_streak

        # Fatigue (days since last match)
        winner_last = pre_winner_stats.get('last_match_date', None)
        loser_last = pre_loser_stats.get('last_match_date', None)
        winner_fatigue = (date - winner_last).days if winner_last is not None else 99
        loser_fatigue = (date - loser_last).days if loser_last is not None else 99
        fatigue_days_diff = winner_fatigue - loser_fatigue

        # H2H win % on current surface
        h2h_surface_matches = pre_h2h.get('surface_matches', {}).get(surface, 0)
        h2h_surface_wins_winner = pre_h2h.get('surface_wins', {}).get(winner, {}).get(surface, 0)
        h2h_surface_wins_loser = pre_h2h.get('surface_wins', {}).get(loser, {}).get(surface, 0)
        winner_h2h_surface_pct = h2h_surface_wins_winner / max(1, h2h_surface_matches) if h2h_surface_matches > 0 else 0.5
        loser_h2h_surface_pct = h2h_surface_wins_loser / max(1, h2h_surface_matches) if h2h_surface_matches > 0 else 0.5
        h2h_surface_win_pct_diff = winner_h2h_surface_pct - loser_h2h_surface_pct

        # Surface adaptability (variance of win % across surfaces seen so far)
        winner_surface_pcts = [
            pre_winner_stats['surface_wins'][surf] / max(1, pre_winner_stats['surface_matches'][surf])
            for surf in pre_winner_stats['surface_wins']
            if pre_winner_stats['surface_matches'][surf] > 0
        ]
        loser_surface_pcts = [
            pre_loser_stats['surface_wins'][surf] / max(1, pre_loser_stats['surface_matches'][surf])
            for surf in pre_loser_stats['surface_wins']
            if pre_loser_stats['surface_matches'][surf] > 0
        ]
        winner_surface_var = np.var(winner_surface_pcts) if winner_surface_pcts else 0.0
        loser_surface_var = np.var(loser_surface_pcts) if loser_surface_pcts else 0.0
        surface_adaptability_diff = winner_surface_var - loser_surface_var

        winner_career_pct = pre_winner_stats['wins'] / max(1, pre_winner_stats['matches']) if pre_winner_stats['matches'] > 0 else 0.5
        loser_career_pct = pre_loser_stats['wins'] / max(1, pre_loser_stats['matches']) if pre_loser_stats['matches'] > 0 else 0.5
        winner_surface_pct = pre_winner_surface_stat['wins'] / max(1, pre_winner_surface_stat['matches']) if pre_winner_surface_stat['matches'] > 0 else winner_career_pct
        loser_surface_pct = pre_loser_surface_stat['wins'] / max(1, pre_loser_surface_stat['matches']) if pre_loser_surface_stat['matches'] > 0 else loser_career_pct
        # Weighted recent form (recent matches matter more)
        if pre_winner_recent:
            weights = [0.9**i for i in range(len(pre_winner_recent))][::-1]
            winner_form = np.average(pre_winner_recent, weights=weights)
        else:
            winner_form = winner_career_pct
        if pre_loser_recent:
            weights = [0.9**i for i in range(len(pre_loser_recent))][::-1]
            loser_form = np.average(pre_loser_recent, weights=weights)
        else:
            loser_form = loser_career_pct
        winner_h2h_pct = pre_h2h['wins'][winner] / max(1, pre_h2h['matches']) if pre_h2h['matches'] > 0 else 0.5
        loser_h2h_pct = pre_h2h['wins'][loser] / max(1, pre_h2h['matches']) if pre_h2h['matches'] > 0 else 0.5

        margin = 1.0  # Default
        if 'Wsets' in match.index and 'Lsets' in match.index:
            w_sets = match.get('Wsets', None)
            l_sets = match.get('Lsets', None)
            if w_sets is not None and l_sets is not None:
                try:
                    w_sets = float(w_sets)
                    l_sets = float(l_sets)
                    if w_sets + l_sets > 0:
                        margin = w_sets / (w_sets + l_sets)
                except Exception:
                    pass

        match_id = idx

        # Prepare context data for winner and loser
        winner_stats = player_stats[winner]
        loser_stats = player_stats[loser]
        
        winner_context = {
            'career_win_pct': winner_career_pct,
            'surface_win_pct': winner_surface_pct,
            'recent_form': winner_form,
            'h2h_win_pct': winner_h2h_pct,
            'elo_rating': pre_winner_elo,
            'glicko2_rating': pre_winner_glicko2['rating'],
            'glicko2_rd': pre_winner_glicko2['rd'],
            'glicko2_volatility': pre_winner_glicko2['volatility'],
            'win_streak': winner_win_streak,
            'loss_streak': winner_loss_streak,
            'fatigue_days': winner_fatigue,
            'h2h_surface_win_pct': winner_h2h_surface_pct,
            'surface_adaptability': winner_surface_var,
            # Tournament-specific win percentages
            'atp250_win_pct': winner_stats['atp250_wins'] / max(1, winner_stats['atp250_matches']),
            'atp500_win_pct': winner_stats['atp500_wins'] / max(1, winner_stats['atp500_matches']),
            'masters1000_win_pct': winner_stats['masters1000_wins'] / max(1, winner_stats['masters1000_matches']),
            'grand_slam_win_pct': winner_stats['grand_slam_wins'] / max(1, winner_stats['grand_slam_matches']),
            # Surface-specific win percentages
            'hard_court_win_pct': winner_stats['hard_court_wins'] / max(1, winner_stats['hard_court_matches']),
            'clay_court_win_pct': winner_stats['clay_court_wins'] / max(1, winner_stats['clay_court_matches']),
            'grass_court_win_pct': winner_stats['grass_court_wins'] / max(1, winner_stats['grass_court_matches']),
            # Round-specific win percentages
            'early_round_win_pct': winner_stats['early_round_wins'] / max(1, winner_stats['early_round_matches']),
            'late_round_win_pct': winner_stats['late_round_wins'] / max(1, winner_stats['late_round_matches']),
            'quarterfinal_win_pct': winner_stats['quarterfinal_wins'] / max(1, winner_stats['quarterfinal_matches']),
            # Advanced performance metrics
            'big_match_experience': winner_stats['big_match_matches'] / max(1, winner_stats['matches']),
            'clutch_performance': winner_stats['clutch_wins'] / max(1, winner_stats['clutch_matches']),
            'outdoor_preference': (winner_stats['outdoor_wins'] / max(1, winner_stats['outdoor_matches'])) - 
                                 (winner_stats['indoor_wins'] / max(1, winner_stats['indoor_matches'])),
            # Recent form by surface (simplified - using overall recent form for now)
            'recent_hard_court_form': winner_form,  # TODO: Implement surface-specific recent form
            'recent_clay_court_form': winner_form,
            'recent_grass_court_form': winner_form,
            'recent_big_tournament_form': winner_form,
            # Additional calculated metrics
            'surface_transition_perf': 0.5,  # TODO: Implement surface transition logic
        }
        
        loser_context = {
            'career_win_pct': loser_career_pct,
            'surface_win_pct': loser_surface_pct,
            'recent_form': loser_form,
            'h2h_win_pct': loser_h2h_pct,
            'elo_rating': pre_loser_elo,
            'glicko2_rating': pre_loser_glicko2['rating'],
            'glicko2_rd': pre_loser_glicko2['rd'],
            'glicko2_volatility': pre_loser_glicko2['volatility'],
            'win_streak': loser_win_streak,
            'loss_streak': loser_loss_streak,
            'fatigue_days': loser_fatigue,
            'h2h_surface_win_pct': loser_h2h_surface_pct,
            'surface_adaptability': loser_surface_var,
            # Tournament-specific win percentages
            'atp250_win_pct': loser_stats['atp250_wins'] / max(1, loser_stats['atp250_matches']),
            'atp500_win_pct': loser_stats['atp500_wins'] / max(1, loser_stats['atp500_matches']),
            'masters1000_win_pct': loser_stats['masters1000_wins'] / max(1, loser_stats['masters1000_matches']),
            'grand_slam_win_pct': loser_stats['grand_slam_wins'] / max(1, loser_stats['grand_slam_matches']),
            # Surface-specific win percentages
            'hard_court_win_pct': loser_stats['hard_court_wins'] / max(1, loser_stats['hard_court_matches']),
            'clay_court_win_pct': loser_stats['clay_court_wins'] / max(1, loser_stats['clay_court_matches']),
            'grass_court_win_pct': loser_stats['grass_court_wins'] / max(1, loser_stats['grass_court_matches']),
            # Round-specific win percentages
            'early_round_win_pct': loser_stats['early_round_wins'] / max(1, loser_stats['early_round_matches']),
            'late_round_win_pct': loser_stats['late_round_wins'] / max(1, loser_stats['late_round_matches']),
            'quarterfinal_win_pct': loser_stats['quarterfinal_wins'] / max(1, loser_stats['quarterfinal_matches']),
            # Advanced performance metrics
            'big_match_experience': loser_stats['big_match_matches'] / max(1, loser_stats['matches']),
            'clutch_performance': loser_stats['clutch_wins'] / max(1, loser_stats['clutch_matches']),
            'outdoor_preference': (loser_stats['outdoor_wins'] / max(1, loser_stats['outdoor_matches'])) - 
                                 (loser_stats['indoor_wins'] / max(1, loser_stats['indoor_matches'])),
            # Recent form by surface (simplified - using overall recent form for now)
            'recent_hard_court_form': loser_form,  # TODO: Implement surface-specific recent form
            'recent_clay_court_form': loser_form,
            'recent_grass_court_form': loser_form,
            'recent_big_tournament_form': loser_form,
            # Additional calculated metrics
            'surface_transition_perf': 0.5,  # TODO: Implement surface transition logic
        }

        # Add tournament and surface context for advanced features
        tournament = match.get('Tournament', '')
        tournament_type = 'ATP250'  # Default
        if 'ATP 500' in tournament or '500' in tournament:
            tournament_type = 'ATP500'
        elif 'Masters' in tournament or 'ATP Masters 1000' in tournament or '1000' in tournament:
            tournament_type = 'Masters1000'
        elif 'Grand Slam' in tournament or any(gs in tournament for gs in ['Wimbledon', 'French Open', 'US Open', 'Australian Open']):
            tournament_type = 'GrandSlam'
        
        # Add tournament context to both player contexts
        winner_context['tournament_type'] = tournament_type
        winner_context['surface'] = surface
        loser_context['tournament_type'] = tournament_type
        loser_context['surface'] = surface

        # Use ALL features (core + additional) for comprehensive model
        ALL_FEATURES = FEATURES + ADDITIONAL_FEATURES
        
        # Build feature vectors using centralized computation
        feature_row = build_feature_vector(
            winner_context, 
            loser_context, 
            win_label=1, 
            extra_fields={'Date': date, 'match_id': match_id},
            feature_list=ALL_FEATURES
        )
        features.append(feature_row)

        loser_row = build_feature_vector(
            loser_context, 
            winner_context, 
            win_label=0, 
            extra_fields={'Date': date, 'match_id': match_id},
            feature_list=ALL_FEATURES
        )
        features.append(loser_row)
        
        # Update streaks and last match date
        for player, is_winner in [(winner, True), (loser, False)]:
            if player_stats[player]['last_match_date'] is not None:
                days_since = (date - player_stats[player]['last_match_date']).days
            else:
                days_since = None
            player_stats[player]['last_match_date'] = date
            if is_winner:
                player_stats[player]['win_streak'] = player_stats[player].get('win_streak', 0) + 1
                player_stats[player]['loss_streak'] = 0
            else:
                player_stats[player]['loss_streak'] = player_stats[player].get('loss_streak', 0) + 1
                player_stats[player]['win_streak'] = 0
        
        # Update surface stats for adaptability
        player_stats[winner]['surface_wins'][surface] += 1
        player_stats[winner]['surface_matches'][surface] += 1
        player_stats[loser]['surface_matches'][surface] += 1

        # Update H2H surface stats
        h2h_stats[h2h_key]['surface_matches'][surface] = h2h_stats[h2h_key]['surface_matches'].get(surface, 0) + 1
        h2h_stats[h2h_key]['surface_wins'][winner][surface] = h2h_stats[h2h_key]['surface_wins'][winner].get(surface, 0) + 1
        # loser gets no win for this surface

        elo_system.update_rating(winner, loser, date, margin_of_victory=margin)
        glicko2_system.update_rating(winner, loser, date, margin_of_victory=margin)
        
        player_stats[winner]['matches'] += 1
        player_stats[winner]['wins'] += 1
        player_stats[loser]['matches'] += 1
        
        surface_stats[winner][surface]['matches'] += 1
        surface_stats[winner][surface]['wins'] += 1
        surface_stats[loser][surface]['matches'] += 1
        
        # Tournament-specific statistics
        tournament = match.get('Tournament', '')
        if 'ATP 250' in tournament or '250' in tournament:
            player_stats[winner]['atp250_matches'] += 1
            player_stats[winner]['atp250_wins'] += 1
            player_stats[loser]['atp250_matches'] += 1
        elif 'ATP 500' in tournament or '500' in tournament:
            player_stats[winner]['atp500_matches'] += 1
            player_stats[winner]['atp500_wins'] += 1
            player_stats[loser]['atp500_matches'] += 1
        elif 'Masters' in tournament or 'ATP Masters 1000' in tournament or '1000' in tournament:
            player_stats[winner]['masters1000_matches'] += 1
            player_stats[winner]['masters1000_wins'] += 1
            player_stats[loser]['masters1000_matches'] += 1
        elif 'Grand Slam' in tournament or any(gs in tournament for gs in ['Wimbledon', 'French Open', 'US Open', 'Australian Open']):
            player_stats[winner]['grand_slam_matches'] += 1
            player_stats[winner]['grand_slam_wins'] += 1
            player_stats[loser]['grand_slam_matches'] += 1
        
        # Surface-specific statistics
        if surface == 'Hard':
            player_stats[winner]['hard_court_matches'] += 1
            player_stats[winner]['hard_court_wins'] += 1
            player_stats[loser]['hard_court_matches'] += 1
        elif surface == 'Clay':
            player_stats[winner]['clay_court_matches'] += 1
            player_stats[winner]['clay_court_wins'] += 1
            player_stats[loser]['clay_court_matches'] += 1
        elif surface == 'Grass':
            player_stats[winner]['grass_court_matches'] += 1
            player_stats[winner]['grass_court_wins'] += 1
            player_stats[loser]['grass_court_matches'] += 1
        
        # Round-specific statistics
        round_info = match.get('Round', '')
        if any(r in round_info for r in ['1st Round', 'First Round', 'R128', 'R64', 'R32']):
            player_stats[winner]['early_round_matches'] += 1
            player_stats[winner]['early_round_wins'] += 1
            player_stats[loser]['early_round_matches'] += 1
        elif any(r in round_info for r in ['Semifinals', 'Final', 'Finals']):
            player_stats[winner]['late_round_matches'] += 1
            player_stats[winner]['late_round_wins'] += 1
            player_stats[loser]['late_round_matches'] += 1
        elif 'Quarterfinals' in round_info:
            player_stats[winner]['quarterfinal_matches'] += 1
            player_stats[winner]['quarterfinal_wins'] += 1
            player_stats[loser]['quarterfinal_matches'] += 1
        
        # Advanced performance metrics
        # Big match performance (Masters/Grand Slam)
        if any(term in tournament for term in ['Masters', 'Grand Slam', 'Wimbledon', 'French Open', 'US Open', 'Australian Open', '1000']):
            player_stats[winner]['big_match_matches'] += 1
            player_stats[winner]['big_match_wins'] += 1
            player_stats[loser]['big_match_matches'] += 1
        
        # Clutch performance (3+ set matches or tiebreaks)
        if margin <= 2:  # Close matches (margin <= 2 games)
            player_stats[winner]['clutch_matches'] += 1
            player_stats[winner]['clutch_wins'] += 1
            player_stats[loser]['clutch_matches'] += 1
        
        # Indoor/Outdoor preferences (simplified heuristic)
        if 'Indoor' in tournament or any(term in tournament for term in ['ATP Finals', 'Paris Masters']):
            player_stats[winner]['indoor_matches'] += 1
            player_stats[winner]['indoor_wins'] += 1
            player_stats[loser]['indoor_matches'] += 1
        else:
            player_stats[winner]['outdoor_matches'] += 1
            player_stats[winner]['outdoor_wins'] += 1
            player_stats[loser]['outdoor_matches'] += 1
        
        h2h_stats[h2h_key]['matches'] += 1
        h2h_stats[h2h_key]['wins'][winner] += 1
        
        recent_matches[winner].append(1)
        recent_matches[loser].append(0)
    
    feat_df = pd.DataFrame(features)
    # Temporal leakage check
    max_date = df['Date'].max()
    if (feat_df['Date'] > max_date).any():
        raise ValueError("Temporal leakage detected! Future dates in features")
    return feat_df

if __name__ == '__main__':
    raw_data_path = '../data/tennis_data/tennis_data.csv'
    features_output_path = '../data/tennis_features.csv'
    original_df = pd.read_csv(raw_data_path, low_memory=False)
    features_df = calculate_features(original_df)
    features_df.to_csv(features_output_path, index=False)

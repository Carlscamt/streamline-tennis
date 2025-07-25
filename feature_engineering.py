import pandas as pd
import numpy as np
import math
from typing import Dict, Optional
from tqdm import tqdm

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
            
        new_volatility = math.exp(C / 2)
        
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
    
    features = []
    
    for idx, match in tqdm(df.iterrows(), total=len(df)):
        date = match['Date']
        winner = match['Winner']
        loser = match['Loser']
        surface = match['Surface']
        
        for player in [winner, loser]:
            if player not in player_stats:
                player_stats[player] = {'matches': 0, 'wins': 0}
            if player not in surface_stats:
                surface_stats[player] = {surf: {'matches': 0, 'wins': 0} for surf in df['Surface'].unique()}
            if player not in recent_matches:
                recent_matches[player] = []
                
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
            h2h_stats[h2h_key] = {'matches': 0, 'wins': {winner: 0, loser: 0}}
        pre_h2h = h2h_stats[h2h_key].copy()

        winner_career_pct = pre_winner_stats['wins'] / max(1, pre_winner_stats['matches']) if pre_winner_stats['matches'] > 0 else 0.5
        loser_career_pct = pre_loser_stats['wins'] / max(1, pre_loser_stats['matches']) if pre_loser_stats['matches'] > 0 else 0.5
        winner_surface_pct = pre_winner_surface_stat['wins'] / max(1, pre_winner_surface_stat['matches']) if pre_winner_surface_stat['matches'] > 0 else winner_career_pct
        loser_surface_pct = pre_loser_surface_stat['wins'] / max(1, pre_loser_surface_stat['matches']) if pre_loser_surface_stat['matches'] > 0 else loser_career_pct
        winner_form = sum(pre_winner_recent) / len(pre_winner_recent) if pre_winner_recent else winner_career_pct
        loser_form = sum(pre_loser_recent) / len(pre_loser_recent) if pre_loser_recent else loser_career_pct
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

        feature_row = {
            'career_win_pct_diff': winner_career_pct - loser_career_pct,
            'surface_win_pct_diff': winner_surface_pct - loser_surface_pct,
            'recent_form_diff': winner_form - loser_form,
            'h2h_win_pct_diff': winner_h2h_pct - loser_h2h_pct,
            'elo_rating_diff': pre_winner_elo - pre_loser_elo,
            'glicko2_rating_diff': pre_winner_glicko2['rating'] - pre_loser_glicko2['rating'],
            'glicko2_rd_diff': pre_loser_glicko2['rd'] - pre_winner_glicko2['rd'],
            'glicko2_volatility_diff': pre_winner_glicko2['volatility'] - pre_loser_glicko2['volatility'],
            'Win': 1,
            'Date': date,
            'match_id': match_id
        }
        for k in feature_row:
            if k not in ['Date']:
                feature_row[k] = np.nan_to_num(feature_row[k], nan=0.0)
        features.append(feature_row)

        loser_row = {
            'career_win_pct_diff': loser_career_pct - winner_career_pct,
            'surface_win_pct_diff': loser_surface_pct - winner_surface_pct,
            'recent_form_diff': loser_form - winner_form,
            'h2h_win_pct_diff': loser_h2h_pct - winner_h2h_pct,
            'elo_rating_diff': pre_loser_elo - pre_winner_elo,
            'glicko2_rating_diff': pre_loser_glicko2['rating'] - pre_winner_glicko2['rating'],
            'glicko2_rd_diff': pre_winner_glicko2['rd'] - pre_loser_glicko2['rd'],
            'glicko2_volatility_diff': pre_winner_glicko2['volatility'] - pre_loser_glicko2['volatility'],
            'Win': 0,
            'Date': date,
            'match_id': match_id
        }
        for k in loser_row:
            if k not in ['Date']:
                loser_row[k] = np.nan_to_num(loser_row[k], nan=0.0)
        features.append(loser_row)
        
        elo_system.update_rating(winner, loser, date, margin_of_victory=margin)
        glicko2_system.update_rating(winner, loser, date, margin_of_victory=margin)
        
        player_stats[winner]['matches'] += 1
        player_stats[winner]['wins'] += 1
        player_stats[loser]['matches'] += 1
        
        surface_stats[winner][surface]['matches'] += 1
        surface_stats[winner][surface]['wins'] += 1
        surface_stats[loser][surface]['matches'] += 1
        
        h2h_stats[h2h_key]['matches'] += 1
        h2h_stats[h2h_key]['wins'][winner] += 1
        
        recent_matches[winner].append(1)
        recent_matches[loser].append(0)
    
    return pd.DataFrame(features)

if __name__ == '__main__':
    raw_data_path = 'TennisMatch/tennis_data/tennis_data.csv'
    features_output_path = 'tennis_features.csv'
    original_df = pd.read_csv(raw_data_path, low_memory=False)
    features_df = calculate_features(original_df)
    features_df.to_csv(features_output_path, index=False)

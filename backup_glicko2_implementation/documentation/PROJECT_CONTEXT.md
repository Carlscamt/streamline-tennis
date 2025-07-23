# Tennis Predictor App - Project Context

## Overview
This is a comprehensive tennis match prediction and betting advisory system that uses machine learning to predict tennis match outcomes and provides intelligent betting recommendations. The app is designed to help users make informed betting decisions based on statistical analysis and historical data.

## Primary Goals
1. **Match Outcome Prediction**: Predict the winner of tennis matches using machine learning models
2. **Betting Advisory**: Provide intelligent betting recommendations with risk management
3. **Historical Analysis**: Backtest strategies and analyze performance over time
4. **Live Data Integration**: Scrape upcoming tennis matches for real-time predictions

## Core Components

### 1. Data Management
- **Historical Data**: Tennis match data from 2000-2025 stored in Excel/CSV format
- **Data Processing**: Clean and prepare historical tennis match data
- **Feature Engineering**: Advanced statistical features including ELO ratings, win percentages, and head-to-head records

### 2. Machine Learning Model
- **Algorithm**: XGBoost classifier with time-series cross-validation
- **Features**:
  - Career win percentage difference
  - Surface-specific win percentage difference
  - Recent form difference (last 10 matches)
  - Head-to-head win percentage difference
  - Ranking difference
  - ELO rating difference
  - Glicko-2 rating difference
  - Glicko-2 rating deviation difference (uncertainty measure)
  - Glicko-2 volatility difference (consistency measure)
- **Model File**: `tennis_model_fixed.joblib`
- **Surface Encoder**: `surface_encoder.joblib`

### 3. Prediction System
- **ELO Rating System**: Dynamic player rating system with temporal decay
- **Glicko-2 Rating System**: Advanced rating system with uncertainty and volatility measures
- **Surface Specialization**: Surface-specific performance analysis (Hard, Clay, Grass)
- **Recent Form Analysis**: Weight recent matches more heavily
- **Head-to-Head Records**: Historical matchup analysis

### 4. Betting Strategy
- **Kelly Criterion**: Mathematical approach to bet sizing
- **Uncertainty Management**: Shrinkage factor for uncertain predictions
- **Risk Management**: Maximum bet size limits and minimum probability thresholds
- **Bankroll Management**: Track and manage betting capital

### 5. User Interface
- **Streamlit App**: Web-based interface for match predictions
- **Match Predictor**: `match_predictor.py` - Main application interface
- **Real-time Predictions**: Input player names, surface, and date for predictions

### 6. Data Collection
- **Event Scraper**: `event_scraper.py` - Scrapes upcoming matches from SofaScore
- **Filtering**: Focuses on ATP singles matches, filters out ITF/WTA/Junior events

## Key Files and Their Purpose

### Core Application Files
- `match_predictor.py` - Main Streamlit web application for predictions
- `train_model_fixed.py` - Model training with time-series validation
- `feature_engineering_fixed.py` - Feature calculation and ELO rating system
- `betting_strategy_uncertainty.py` - Advanced betting strategy with uncertainty handling
- `event_scraper.py` - Live match data scraping

### Backtesting and Analysis
- `backtest_2025.py` - Backtest model performance on 2025 data
- `backtest_2025_uncertainty.py` - Backtest with uncertainty-aware betting
- `backtesting_engine_optimized.py` - Performance analysis framework

### Data Files
- `tennis_data/` - Historical match data (2000-2025)
- `tennis_data.csv` - Processed historical data
- `tennis_model_fixed.joblib` - Trained machine learning model
- `surface_encoder.joblib` - Surface encoding for model

## Technical Architecture

### Machine Learning Pipeline
1. **Data Loading**: Load historical match data from CSV/Excel files
2. **Feature Engineering**: Calculate advanced statistical features for each match
3. **Model Training**: Train XGBoost classifier with time-series cross-validation
4. **Prediction**: Generate win probabilities for new matches
5. **Betting Advice**: Calculate optimal bet sizes using Kelly Criterion

### ELO Rating System
- **Initial Rating**: 1500 for all players
- **K-Factor**: 30-32 for rating updates
- **Temporal Decay**: Ratings decay over time (0.95-0.97 decay factor)
- **Surface-Specific**: Separate ratings for different court surfaces
- **Margin of Victory**: Considers set scores for rating adjustments

### Glicko-2 Rating System
- **Initial Rating**: 1500 for all players (consistent with ELO)
- **Initial RD**: 350 (rating deviation - uncertainty measure)
- **Initial Volatility**: 0.06 (expected rating fluctuation)
- **System Constant (τ)**: 0.3 (controls volatility changes)
- **Surface-Specific**: Separate ratings for different court surfaces
- **Uncertainty Handling**: RD increases with inactivity, decreases with match play
- **Volatility Tracking**: Measures consistency of performance over time

### Betting Strategy Features
- **Kelly Criterion**: Optimal bet sizing based on edge and odds
- **Uncertainty Shrinkage**: Reduce bet sizes for uncertain predictions
- **Risk Limits**: Maximum 10% of bankroll per bet
- **Minimum Edge**: Only bet when win probability > 55%
- **Fair Odds Calculation**: Compare model odds with market odds

## Model Performance Metrics
- **Accuracy**: Target >53% for profitable betting
- **Cross-Validation**: Time-series split validation
- **Feature Importance**: Ranking difference typically most important
- **Backtesting**: Tested on out-of-sample 2025 data

## Current Status (July 2025)
- Model trained on data through 2024
- Backtesting completed on 2025 data
- Web interface functional
- Live data scraping operational
- Betting strategy implemented with uncertainty handling

## Usage Workflow
1. **Historical Analysis**: Run backtesting to evaluate strategy performance
2. **Model Training**: Retrain model with latest data as needed
3. **Live Predictions**: Use web app to predict upcoming matches
4. **Betting Decisions**: Follow betting recommendations with proper bankroll management
5. **Performance Tracking**: Monitor actual vs predicted results

## Risk Management
- **Bankroll Limits**: Never bet more than 10% of total bankroll
- **Minimum Edge**: Only bet when model shows significant advantage
- **Uncertainty Handling**: Reduce bets when prediction confidence is low
- **Diversification**: Spread bets across multiple matches when possible

## Future Enhancements
- Real-time odds integration
- More sophisticated player injury/form analysis
- Weather and court speed factors
- Enhanced live data sources
- Mobile-responsive interface improvements

## Dependencies
- Python 3.7+
- XGBoost for machine learning
- Streamlit for web interface
- Pandas/NumPy for data processing
- Playwright for web scraping
- Joblib for model serialization

This system represents a complete end-to-end solution for tennis betting analysis, combining statistical modeling, risk management, and user-friendly interfaces to provide actionable betting insights.

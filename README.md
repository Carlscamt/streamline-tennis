# 🎾 Tennis Match Prediction System

A comprehensive tennis match prediction system using machine learning and advanced rating algorithms. Predicts match outcomes with 65% accuracy using ELO and Glicko-2 rating systems.

## 🎯 **Project Status: PRODUCTION READY**
- **Data Leakage**: ✅ Leak-free, match-level split, balanced perspectives (July 2025 cleanup)
- **Glicko-2 Integration**: ✅ Working (26.97% total importance)
- **Model Accuracy**: ✅ 66.5% with robust validation
- **Web Interface**: ✅ Streamlit app functional
- **Code Quality**: ✅ Ultra-clean, simplified, and version-controlled (July 2025)

## ✨ Features
- **Leak-free Feature Engineering**: No dyadic leakage, match-level split, balanced classes
- **Real-time Match Prediction**: Web interface for live predictions
- **Advanced Rating Systems**: ELO and Glicko-2 implementations
- **Uncertainty-Based Betting**: Kelly Criterion betting recommendations
- **Historical Data**: 65,715+ tennis matches from 2000-2025
- **Surface Analysis**: Court surface performance tracking

## 📊 **Final Metrics**
- **Files**: 14 total (down from 42+)
- **Python files**: 5 core implementations
- **Code reduction**: 64% while preserving all functionality
- **Training script**: 53% smaller (141 → 67 lines)
- **Cleanup**: July 2025—code, data, and model artifacts cleaned and committed to GitHub

## 🏗️ **Architecture**
```
Core System (5 files):
├── feature_engineering.py  # ELO + Glicko-2 rating systems
├── train_model.py          # Simplified XGBoost training  
├── match_predictor.py      # Streamlit web application
├── betting_strategy.py     # Uncertainty-based recommendations
└── test_ratings.py         # Validation tests

Data & Models (4 files):
├── tennis_features.csv     # 65,715 matches, leak-free
├── tennis_model.joblib     # Trained model
├── surface_encoder.joblib  # Court surface encoder
└── feature_importance.csv  # Feature analysis
```

## 🚀 **Quick Start**
```bash
# Test system
python test_ratings.py

# Generate features (if needed)  
python feature_engineering.py

# Train model
python train_model.py

# Run web app
streamlit run match_predictor.py
```

## 🧠 **Model Features**
1. `elo_rating_diff` (55.32%) - ELO rating differences
2. `glicko2_rd_diff` (22.25%) - Rating uncertainty (Glicko-2)
3. `surface_win_pct_diff` (11.37%) - Surface performance
4. `h2h_win_pct_diff` (3.39%) - Head-to-head record
5. `glicko2_volatility_diff` (2.38%) - Volatility (Glicko-2)
6. `glicko2_rating_diff` (2.34%) - Glicko-2 ratings
7. **Glicko-2 total**: 26.97% contribution

## ⚡ **Performance**
- **Accuracy**: 66.50% (realistic, no data leakage)
- **Training time**: Fast (simplified validation)
- **Memory usage**: Minimal (clean imports)
- **Deployment**: Ready (no dependencies issues)

## 💾 **Backup Available**
Complete original implementation preserved in:
`backup_glicko2_implementation/`

---

**Status**: ✅ **ULTRA-CLEAN, LEAK-FREE & PRODUCTION-READY** 🎾
*Latest update: July 2025—major cleanup, leak-free pipeline, and GitHub version control.*

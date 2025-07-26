# 🎾 Tennis Match Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

**A comprehensive tennis match prediction system using machine learning and advanced rating algorithms. Predicts match outcomes with 65%+ accuracy using ELO and Glicko-2 rating systems.**

## 🚀 **Ultra-Streamlined Feature Engineering**

This system features an **ultra-streamlined workflow** where adding new features requires editing **only one file** (`src/features.py`). The entire pipeline automatically integrates new features without any additional configuration.

### ✨ Quick Feature Addition:
```python
# Edit src/features.py only:
# 1. Add feature name to FEATURES list
# 2. Add computation logic to compute_feature()
# That's it! Feature flows through entire pipeline automatically.
```

## 📁 **Professional Repository Structure**

```
🎾 TennisMatch/
├── 📱 Application
│   ├── match_predictor.py       # 🎮 Streamlit web application
│   └── run.py                   # 🚀 Quick start script
│
├── 🧠 Core System
│   ├── src/
│   │   ├── features.py          # ⭐ Central feature system
│   │   ├── feature_engineering.py # ELO & Glicko-2 ratings
│   │   ├── train_model.py       # XGBoost training
│   │   ├── betting_strategy.py  # Kelly Criterion
│   │   ├── backtest_2025.py     # Validation system
│   │   ├── feature_builder.py   # Compatibility wrapper
│   │   └── __init__.py          # Package initialization
│   │
├── 📊 Data & Models
│   ├── data/
│   │   ├── tennis_features.csv      # 65,715+ match dataset
│   │   ├── feature_importance.csv   # Feature analysis
│   │   └── tennis_data/             # Raw data directory
│   │
│   └── models/
│       └── tennis_model.joblib      # Trained XGBoost model
│
├── 📚 Documentation
│   ├── README.md               # This comprehensive guide
│   ├── CONTRIBUTING.md         # Development guidelines
│   ├── LICENSE                 # MIT license
│   └── docs/                   # Additional documentation
│
└── ⚙️ Configuration
    ├── requirements.txt        # Python dependencies
    ├── .gitignore             # Git ignore rules
    └── .github/               # GitHub workflows
```

## ⚡ **Quick Start**

### **1. Install & Run (3 Commands)**
```bash
# Clone and setup
git clone https://github.com/Carlscamt/TennisMatch.git
cd TennisMatch
pip install -r requirements.txt

# Run the web app
streamlit run match_predictor.py
# 🎮 Access at: http://localhost:8501
```

### **2. Quick Command Interface**
```bash
# Use the convenient run script
python run.py predict     # Start web app
python run.py train       # Train model
python run.py backtest    # Run validation
python run.py features    # Generate features
```

## 🎯 **Key Features**

### 🎮 **Interactive Web Application**
- **Real-time Match Prediction**: Select any two players, get instant win probabilities
- **Surface-Specific Analysis**: Account for different court surfaces (Hard, Clay, Grass, Carpet)
- **Advanced Rating Systems**: Live ELO and Glicko-2 calculations
- **Betting Strategy**: Kelly Criterion-based recommendations with uncertainty shrinkage
- **Historical Context**: Player statistics, head-to-head records, recent form

### 📊 **Machine Learning Pipeline**
- **Leak-free Feature Engineering**: Proper temporal splits, no data leakage
- **Advanced Rating Systems**: ELO and Glicko-2 implementations
- **Feature Categories**: Performance, ratings, head-to-head, temporal, surface analysis
- **Robust Validation**: Walk-forward backtesting across multiple years
- **Model Architecture**: XGBoost classifier optimized for tennis prediction

### 🚀 **Ultra-Streamlined Development**
- **Single-File Feature Addition**: Edit only `src/features.py` for new features
- **Automatic Integration**: Features flow through entire pipeline automatically
- **Centralized Logic**: Single source of truth for all computations
- **Backward Compatible**: Existing code continues to work seamlessly

## 📊 **Feature System**

### **Core Features (8 - used by current model):**
| Feature | Description | Importance |
|---------|-------------|------------|
| `career_win_pct_diff` | Overall career win percentage difference | High |
| `surface_win_pct_diff` | Surface-specific win percentage difference | High |
| `recent_form_diff` | Recent performance difference (last 10 matches) | Medium |
| `h2h_win_pct_diff` | Head-to-head win percentage difference | Medium |
| `elo_rating_diff` | ELO rating difference | High |
| `glicko2_rating_diff` | Glicko-2 rating difference | Very High |
| `glicko2_rd_diff` | Glicko-2 rating deviation difference | Medium |
| `glicko2_volatility_diff` | Glicko-2 volatility difference | Low |

### **Additional Features (5 - require model retraining):**
- `fatigue_days_diff`: Days since last match difference
- `h2h_surface_win_pct_diff`: Head-to-head on current surface
- `surface_adaptability_diff`: Surface adaptation ability
- `win_streak_diff`: Current winning streak difference
- `loss_streak_diff`: Current losing streak difference

## 🔧 **Adding New Features**

The **ultra-streamlined workflow** makes adding features incredibly simple:

```python
# Step 1: Edit src/features.py - Add to ADDITIONAL_FEATURES list
ADDITIONAL_FEATURES = [
    'fatigue_days_diff',
    'win_streak_diff',
    'your_new_feature',  # ← Add here
]

# Step 2: Edit src/features.py - Add computation logic
def compute_feature(feature_name: str, context: Dict[str, Any]) -> float:
    # ... existing features ...
    
    elif feature_name == 'your_new_feature':
        return p1_data['your_stat'] - p2_data['your_stat']  # ← Add here
```

**That's it!** The feature automatically flows through:
- ✅ Training pipeline (`src/train_model.py`)
- ✅ Web application (`match_predictor.py`)
- ✅ Backtesting (`src/backtest_2025.py`)
- ✅ All analysis tools

## 🎮 **Usage Examples**

### **Interactive Web Prediction:**
1. Run `streamlit run match_predictor.py`
2. Select two players from dropdown
3. Choose court surface
4. Get instant win probabilities and betting advice

### **Programmatic Usage:**
```python
from src.features import build_prediction_feature_vector, FEATURES
from src.betting_strategy import UncertaintyShrinkageBetting
import joblib

# Load model
model = joblib.load('models/tennis_model.joblib')

# Build feature vector
features = build_prediction_feature_vector(player1_stats, player2_stats)

# Make prediction
probability = model.predict_proba([features[FEATURES]])[0, 1]

# Get betting recommendation
betting = UncertaintyShrinkageBetting()
bet_size, odds = betting.calculate_kelly_fraction(probability, market_odds=2.0)
```

### **Command Line Interface:**
```bash
# Train new model
python run.py train

# Run backtesting
python run.py backtest

# Generate features from raw data
python run.py features
```

## 📈 **Performance Metrics**

- **Model Accuracy**: 66.5% (validated with walk-forward backtesting)
- **Training Dataset**: 65,715+ tennis matches (2000-2025)
- **Feature Count**: 8 core features (13 total available)
- **Data Quality**: Leak-free, match-level splits, balanced classes
- **Validation Method**: Walk-forward temporal validation
- **Feature Importance**: Glicko-2 system accounts for 26.97% of model importance

## 🏗️ **Architecture**

### **Data Flow:**
```
Raw Data → Feature Engineering → Model Training → Web Application
    ↓              ↓                    ↓             ↓
Tennis Data → Rating Systems → XGBoost Model → Streamlit UI
    ↓              ↓                    ↓             ↓
Historical → ELO/Glicko-2 → Predictions → User Interface
```

### **Model Pipeline:**
1. **Data Processing**: Load and preprocess historical match data
2. **Feature Engineering**: Calculate ratings and statistics using `src/features.py`
3. **Model Training**: Train XGBoost classifier with proper temporal splits
4. **Validation**: Walk-forward backtesting across multiple years
5. **Deployment**: Streamlit web application for real-time predictions

### **Rating Systems:**
- **ELO Rating**: Chess-derived system adapted for tennis, surface-specific
- **Glicko-2**: Advanced probabilistic rating with uncertainty quantification

## 🔧 **Technical Requirements**

### **Dependencies:**
- **Python**: 3.8+ (recommended: 3.9+)
- **Core ML**: scikit-learn, XGBoost, pandas, numpy
- **Web Framework**: Streamlit
- **Utilities**: joblib, tqdm

### **System Requirements:**
- **RAM**: 4GB+ (for full dataset processing)
- **Storage**: 500MB+ (for data and models)
- **CPU**: Modern multi-core processor recommended

### **Installation:**
```bash
# Standard installation
pip install -r requirements.txt

# Development installation
pip install -r requirements.txt
pip install -e .  # Editable install for development
```

## 📚 **Documentation**

### **File Documentation:**
Each file contains comprehensive documentation:
- **`src/features.py`**: Central feature system with detailed examples
- **`match_predictor.py`**: Interactive web application guide
- **`src/feature_engineering.py`**: Rating systems and data processing
- **`src/train_model.py`**: Model training pipeline documentation
- **`src/betting_strategy.py`**: Kelly Criterion betting strategy
- **`src/backtest_2025.py`**: Validation system documentation

### **API Reference:**
- **Feature System**: Add features by editing `src/features.py`
- **Model Training**: Use `src/train_model.py` or `python run.py train`
- **Prediction**: Use `match_predictor.py` or import from `src`
- **Validation**: Use `src/backtest_2025.py` or `python run.py backtest`

## 🤝 **Contributing**

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### **Quick Contribution Guide:**
1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/your-feature`
3. **Edit** `src/features.py` to add your feature
4. **Test** with `python run.py train` and `python run.py predict`
5. **Submit** pull request with clear description

### **Contribution Ideas:**
- 🎯 **New Features**: Player rankings, weather data, tournament context
- 🔧 **Improvements**: Model optimization, UI enhancements, performance
- 📚 **Documentation**: Tutorials, examples, API documentation
- 🧪 **Testing**: Unit tests, integration tests, validation

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 **About**

This tennis prediction system combines advanced machine learning with tennis domain expertise to provide accurate match predictions. The ultra-streamlined architecture makes it easy to experiment with new features while maintaining production-quality code.

**Perfect for:**
- 🎾 **Tennis Analysts**: Professional match analysis and prediction
- 📊 **Data Scientists**: Sports analytics and machine learning research
- 🎮 **Developers**: Clean, well-documented codebase for extension
- 🎯 **Enthusiasts**: User-friendly web interface for match predictions

---

**🚀 Start predicting tennis matches with confidence!** 🎾✨

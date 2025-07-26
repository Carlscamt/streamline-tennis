# 🤝 Contributing to Tennis Match Prediction System

Thank you for your interest in contributing to the Tennis Match Prediction System! This guide will help you get started.

## 🚀 Quick Start for Contributors

### 1. **Fork and Clone**
```bash
git clone https://github.com/yourusername/TennisMatch.git
cd TennisMatch
```

### 2. **Set Up Environment**
```bash
pip install -r requirements.txt
```

### 3. **Verify Setup**
```bash
python run.py predict  # Should start the web app
```

## 🎯 **Ultra-Streamlined Feature Development**

The beauty of this system is that **adding features requires editing only ONE file**!

### **Adding New Features (Ultra-Simple):**

1. **Edit `src/features.py`** - Add to ADDITIONAL_FEATURES list:
```python
ADDITIONAL_FEATURES = [
    'fatigue_days_diff',
    'win_streak_diff', 
    'your_new_feature',  # ← Add here
]
```

2. **Edit `src/features.py`** - Add computation logic:
```python
def compute_feature(feature_name: str, context: Dict[str, Any]) -> float:
    # ... existing features ...
    
    elif feature_name == 'your_new_feature':
        return p1_data['your_stat'] - p2_data['your_stat']  # ← Add here
```

3. **That's it!** Your feature automatically flows through:
   - ✅ Training pipeline
   - ✅ Web application  
   - ✅ Backtesting
   - ✅ All analysis tools

## 📊 **Types of Contributions Welcome**

### **🎮 New Features**
- **Player Statistics**: Age, ranking, prize money, etc.
- **Match Context**: Tournament level, stakes, weather
- **Advanced Metrics**: Serve speed, shot patterns, momentum
- **Temporal Analysis**: Seasonal effects, career phases

### **🔧 System Improvements**
- **Model Enhancements**: New algorithms, hyperparameter tuning
- **Data Sources**: Integration with additional tennis databases
- **UI/UX**: Streamlit interface improvements
- **Performance**: Speed optimizations, caching

### **📚 Documentation**
- **Tutorials**: Step-by-step guides for new users
- **Examples**: Real-world usage scenarios
- **API Documentation**: Function and class documentation
- **Research**: Analysis of feature importance, model behavior

### **🧪 Testing & Validation**
- **Unit Tests**: Component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Benchmarks**: Speed and accuracy metrics
- **Data Validation**: Input data quality checks

## 🏗️ **Project Structure**

Understanding the codebase:

```
TennisMatch/
├── 📊 src/                    # Core system
│   ├── features.py            # ⭐ Central feature system
│   ├── feature_engineering.py # Rating systems  
│   ├── train_model.py         # Model training
│   ├── betting_strategy.py    # Betting logic
│   └── backtest_2025.py       # Validation
│
├── 📱 Web App
│   └── match_predictor.py     # Streamlit interface
│
├── 📁 data/                   # Datasets
├── 🤖 models/                 # Trained models
└── 📚 docs/                   # Documentation
```

## 🔄 **Development Workflow**

### **1. Feature Development**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Edit src/features.py only
# Add feature name + computation logic

# Test locally
python run.py train
python run.py predict

# Commit and push
git commit -m "Add your_feature_name for player analysis"
git push origin feature/your-feature-name
```

### **2. Testing Your Changes**
```bash
# Run backtesting to validate
python run.py backtest

# Start web app to test UI
python run.py predict

# Check all features work
python -c "from src.features import FEATURES; print(len(FEATURES), 'features loaded')"
```

### **3. Submitting Changes**
1. **Create Pull Request** with clear description
2. **Include Test Results** (accuracy, performance)
3. **Document New Features** in PR description
4. **Address Review Feedback** promptly

## 📏 **Code Standards**

### **Python Style**
- Follow PEP 8 style guidelines
- Use type hints where helpful
- Add docstrings for new functions
- Keep functions focused and small

### **Feature Implementation**
- **Vectorized Operations**: Use pandas/numpy efficiently
- **Handle Missing Data**: Use np.nan_to_num() or similar
- **Meaningful Names**: Clear, descriptive feature names
- **Documentation**: Comment complex calculations

### **Example Good Feature:**
```python
elif feature_name == 'ranking_momentum':
    """
    Calculates difference in ranking momentum (trend direction).
    Positive values favor player 1.
    """
    p1_momentum = calculate_ranking_trend(p1_data['recent_rankings'])
    p2_momentum = calculate_ranking_trend(p2_data['recent_rankings'])
    return p1_momentum - p2_momentum
```

## 🧪 **Testing Guidelines**

### **Feature Testing**
Before submitting new features:

1. **Computation Test**: Ensure no errors/NaN values
2. **Integration Test**: Verify works in full pipeline  
3. **Performance Test**: Check impact on training/prediction speed
4. **Validation Test**: Run backtesting with new feature

### **Manual Testing Checklist**
- [ ] Feature computes without errors
- [ ] Web app loads and works
- [ ] Training completes successfully
- [ ] Backtesting runs without issues
- [ ] Documentation is clear

## 💡 **Getting Help**

### **Common Issues**

**"Import Error"**: Check you're in the right directory and virtual environment

**"Feature Not Found"**: Ensure feature name is in FEATURES or ADDITIONAL_FEATURES list

**"Model Mismatch"**: New features require model retraining

**"Data Missing"**: Check data file paths and availability

### **Resources**
- **README.md**: Complete system overview
- **Code Documentation**: Detailed docstrings in each file
- **Issues**: GitHub issues for bug reports and feature requests

## 🎯 **Contribution Ideas**

### **Beginner-Friendly**
- Add simple statistical features (age difference, height difference)
- Improve documentation and examples
- Create tutorials for specific use cases
- Test and report bugs

### **Intermediate**
- Implement new rating systems (TrueSkill, Bradley-Terry)
- Add tournament context features
- Optimize performance bottlenecks
- Create visualization tools

### **Advanced**
- Integrate external data sources (weather, odds)
- Implement neural network models
- Create real-time data pipelines
- Build mobile app interface

## 🏆 **Recognition**

Contributors will be:
- **Credited** in release notes
- **Listed** in contributors section
- **Acknowledged** for significant contributions
- **Invited** to maintainer team for sustained contributions

## 📝 **License**

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping make tennis prediction more accurate and accessible!** 🎾✨

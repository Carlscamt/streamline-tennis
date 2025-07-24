import pandas as pd
from autogluon.tabular import TabularPredictor
import importlib

def print_gpu_info():
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            print(f"Total memory: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory // (1024**2)} MB")
            print(f"Allocated memory: {torch.cuda.memory_allocated() // (1024**2)} MB")
            print(f"Reserved memory: {torch.cuda.memory_reserved() // (1024**2)} MB")
        else:
            print("PyTorch CUDA not available.")
    else:
        print("PyTorch not installed.")

def train_model_v1_1():
    print("Loading data...")
    print("\n--- GPU Info Before Training ---")
    print("\n==============================")
    print("  GPU Info Before Training")
    print("==============================")
    print_gpu_info()
    df = pd.read_csv('tennis_features_v2.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    for col in ['Tournament', 'Round', 'Surface', 'player1_hand_encoded', 'player2_hand_encoded']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    features = [
        'career_win_pct_diff', 'surface_win_pct_diff', 'recent_form_diff',
        'h2h_win_pct_diff', 'elo_rating_diff', 'glicko2_rating_diff', 'glicko2_rd_diff',
        'player1_Hard_win_pct', 'player2_Hard_win_pct',
        'player1_Clay_win_pct', 'player2_Clay_win_pct',
        'player1_Grass_win_pct', 'player2_Grass_win_pct',
        'player1_recent_win_pct', 'player2_recent_win_pct',
        'player1_matches_14d', 'player2_matches_14d',
        'round_encoded', 'player1_aces', 'player2_aces',
        'player1_double_faults', 'player2_double_faults',
        'surface_form_interaction', 'player1_hand_encoded', 'player2_hand_encoded'
    ]
    label = 'Win'
    print(f"Dataset: {len(df)} matches, {len(features)} features")
    print(f"\n==============================")
    print(f"  Data Summary")
    print(f"==============================")
    print(f"Dataset: {len(df)} matches, {len(features)} features")
    split_date = df['Date'].quantile(0.8)
    train_data = df[df['Date'] <= split_date]
    test_data = df[df['Date'] > split_date]
    print(f"Training: {len(train_data)} samples")
    print(f"Testing: {len(test_data)} samples")
    unique_labels = train_data[label].nunique()
    print(f"Unique values in label column '{label}': {train_data[label].unique()}")
    if unique_labels < 2:
        raise ValueError(f"Label column '{label}' must have at least two unique values. Found {unique_labels}.")
    print(f"Train 'Win' value counts: {train_data[label].value_counts().to_dict()}")
    print(f"Train 'Win' value counts: {train_data[label].value_counts().to_dict()}")
    print(f"Test 'Win' value counts: {test_data[label].value_counts().to_dict()}")
    print(f"Full dataset 'Win' value counts: {df[label].value_counts().to_dict()}")
    print(f"Full dataset 'Win' value counts: {df[label].value_counts().to_dict()}")
    hyperparameters = {
        'NN_TORCH': {'ag_args_fit': {'num_gpus': 1}},
        'CAT': {'ag_args_fit': {'num_gpus': 1}},
        'XGB': {'ag_args_fit': {'num_gpus': 1}},
        'FASTAI': {'ag_args_fit': {'num_gpus': 1}},
        'GBM': [{'ag_args_fit': {'num_gpus': 1}}],
    }
    predictor = TabularPredictor(label=label, eval_metric='roc_auc').fit(
        train_data[features + [label]],
        presets='good',
        time_limit=1800,
        num_gpus=1,
        num_cpus=1,
        hyperparameters=hyperparameters,
        auto_stack=False,
        dynamic_stacking=False
    )
    print("\n==============================")
    print("  GPU Info After Training")
    print("==============================")
    print_gpu_info()
    print_gpu_info()
    predictions = predictor.predict(test_data[features])
    accuracy = (predictions == test_data[label]).mean()
    print("\n==============================")
    print("  Model Performance Summary")
    print("==============================")
    print(f"Accuracy on test set: {accuracy:.4f}")
    print(f"Best model: {predictor.model_best}")
    print(f"\nAccuracy: {accuracy:.4f}")
    predictor.save('autogluon_model')
    print("\nPredictor saved as 'autogluon_model'!")

if __name__ == "__main__":
    train_model_v1_1()

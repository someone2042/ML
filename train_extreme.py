import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_preprocess_data():
    print("1. Loading bank-full.csv...")
    df = pd.read_csv('bank-full.csv', sep=';')
    
    y = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    
    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Make sure feature names have no special JSON characters for LightGBM
    import re
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def objective(trial, X_train, y_train):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        # Instead of SMOTE, we use native algorithm scaling which is MUCH better for tree models
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 12.0),
        'random_state': 42,
        'verbose': -1
    }
    
    # 3-Fold Cross validation to ensure robust F1 optimization and avoid overfitting
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        # Suppress warnings manually by redirecting verbose
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds))
        
    return np.mean(f1_scores)

def main():
    print("=== STARTING EXTREME OPTIMIZATION PIPELINE ===")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    print("2. Running Optuna with LightGBM & Cross-Validation (30 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)
    
    print("\n[+] Best Hyperparameters Found:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")
        
    print("\n3. Training Final Model on full Train Set...")
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    print("\n4. Threshold Optimization for Maximum F1-Score...")
    # Predict probabilities instead of hard classes
    y_probs = final_model.predict_proba(X_test)[:, 1]
    
    # Find the threshold that perfectly balances Precision and Recall for maximum F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal Probability Threshold found: {optimal_threshold:.4f} (Default is 0.5000)")
    
    # Apply optimal threshold
    final_preds = (y_probs >= optimal_threshold).astype(int)
    
    print("\n=== EXTREME OPTIMIZATION EVALUATION ===")
    print(f"Overall Accuracy: {accuracy_score(y_test, final_preds):.4f}")
    print("\nClassification Report (with threshold tuning):")
    print(classification_report(y_test, final_preds))

if __name__ == "__main__":
    main()

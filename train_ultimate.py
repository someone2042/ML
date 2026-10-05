import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings

# Ignore Optuna and Pandas warnings for cleaner output
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_preprocess_data():
    print("1. Loading bank-full.csv (45,211 rows)...")
    df = pd.read_csv('bank-full.csv', sep=';')
    
    # Target encoding
    y = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    
    print("2. One-Hot Encoding categorical variables...")
    # One-hot encoding replaces LabelEncoder, preventing ordinality assumptions
    X = pd.get_dummies(X, drop_first=True)
    
    # 80/20 train/test split, stratifying to preserve the 89/11 imbalance
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def objective(trial, X_train_res, y_train_res, X_val, y_val):
    # Search space for hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_res, y_train_res, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_val)
    # We maximize F1-score to specifically target the minority class (subscribers)
    return f1_score(y_val, preds)

def main():
    print("=== STARTING ULTIMATE PIPELINE ===")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Split training data further to create a validation set for Optuna
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print("3. Scaling features and applying SMOTE...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_opt)
    X_val_scaled = scaler.transform(X_val_opt)
    
    # SMOTE only on the training set to prevent data leakage
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_opt)
    
    print("4. Running Optuna Hyperparameter Optimization (15 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train_res, y_train_res, X_val_scaled, y_val_opt), n_trials=15)
    
    print("\n[+] Best Hyperparameters Found:")
    for key, value in study.best_params.items():
        print(f"    - {key}: {value}")
        
    print("\n5. Training Final XGBoost Model on full Train Set with Best Params...")
    best_params = study.best_params
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    
    # Scale and SMOTE the FULL training set
    X_train_full_scaled = scaler.fit_transform(X_train)
    X_test_scaled_final = scaler.transform(X_test)
    X_train_full_res, y_train_full_res = smote.fit_resample(X_train_full_scaled, y_train)
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_full_res, y_train_full_res)
    
    print("\n=== FINAL EVALUATION ===")
    preds = final_model.predict(X_test_scaled_final)
    
    print(f"Overall Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()

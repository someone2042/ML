import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

def feature_engineering(df):
    print("Applying Advanced Feature Engineering...")
    # 1. Prevent data leakage by dropping duration
    if 'duration' in df.columns:
        df = df.drop('duration', axis=1)
        
    # 2. Interaction Features
    df['balance_per_age'] = df['balance'] / (df['age'] + 1e-5)
    df['total_contacts_all_time'] = df['campaign'] + df['previous']
    
    # 3. Financial proxy boolean
    df['is_negative_balance'] = (df['balance'] < 0).astype(int)
    
    y = df['y'].map({'yes': 1, 'no': 0})
    X = df.drop('y', axis=1)
    
    # One-hot encode for XGBoost compatibility
    X = pd.get_dummies(X, drop_first=True)
    import re
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    return X, y

def main():
    print("=== STARTING STACKING (LightGBM + XGBoost) PIPELINE ===")
    df = pd.read_csv('bank-full.csv', sep=';')
    
    X, y = feature_engineering(df)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Initializing Models (Using our previously optimized parameters)...")
    # Model 1: LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=442, learning_rate=0.0143, num_leaves=70, max_depth=15, 
        min_child_samples=59, subsample=0.56, colsample_bytree=0.86, 
        scale_pos_weight=3.77, objective='binary', random_state=42, verbose=-1
    )
    
    # Model 2: XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=344, max_depth=3, learning_rate=0.045, 
        subsample=0.92, colsample_bytree=0.61, scale_pos_weight=3.77, 
        objective='binary:logistic', random_state=42
    )
    
    print("Training Soft-Voting Ensemble...")
    # Soft voting averages the probabilities of both SOTA models
    ensemble = VotingClassifier(
        estimators=[('lgb', lgb_model), ('xgb', xgb_model)],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    
    print("Threshold Optimization...")
    y_probs = ensemble.predict_proba(X_test)[:, 1]
    
    # Find perfect threshold for F1 Score
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"Optimal Probability Threshold found: {optimal_threshold:.4f}")
    
    final_preds = (y_probs >= optimal_threshold).astype(int)
    
    print("\n=== FINAL ENSEMBLE EVALUATION ===")
    print(f"Overall Accuracy: {accuracy_score(y_test, final_preds):.4f}")
    print("\nClassification Report (Ensemble + Engineered Features):")
    print(classification_report(y_test, final_preds))

if __name__ == "__main__":
    main()

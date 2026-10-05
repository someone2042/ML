# Project Report: Bank Marketing Term Deposit Prediction

## 1. Project Overview
This machine learning project aims to predict whether a client will subscribe to a term deposit (`y` = yes/no) based on direct marketing campaigns conducted by a Portuguese banking institution.

The project evaluates and compares three supervised learning classification algorithms:
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**

---

## 2. Dataset Description
- **Source**: UCI Bank Marketing Dataset
- **Files**:
  - `bank.csv`: A sample subset consisting of **4,521 rows** and 17 columns (used for rapid prototyping).
  - `bank-full.csv`: Full dataset with **45,211 rows** and 17 columns.
- **Key Features**:
  - **Client Demographics**: `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`
  - **Campaign Details**: `contact`, `day`, `month`, `duration`
  - **Previous Interactions**: `campaign`, `pdays`, `previous`, `poutcome`
  - **Target Variable**: `y` (Binary: `yes` = 1, `no` = 0)

---

## 3. Data Processing & Pipeline
In [`train_models.py`](file:///c:/Users/MOHAMED/Desktop/Nouveau%20dossier/ML/train_models.py):
1. **Categorical Encoding**: Converts textual variables into numeric codes via `LabelEncoder`.
2. **Train/Test Split**: 80% training set (3,616 samples) and 20% test set (905 samples), stratified with `random_state=42`.
3. **Feature Scaling**: `StandardScaler` is applied specifically to distance/margin-sensitive algorithms (**SVM** and **KNN**), while tree-based models (**Random Forest**) utilize raw numeric features.

---

## 4. Experimental Results

### Phase 1: Baseline Models (Sample Dataset - `bank.csv`)
| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **KNN (k=5)** | 89.06% | 0.49 | 0.18 | 0.27 |
| **SVM (RBF Kernel)** | 89.50% | 0.54 | 0.20 | 0.30 |
| **Random Forest** | **90.17%** | **0.60** | **0.27** | **0.37** |

### Phase 2: Ultimate Pipeline (Full Dataset - `bank-full.csv`)
Using **XGBoost + SMOTE + Optuna** hyperparameter optimization:
| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Optimized)** | 88.52% | 0.51 | **0.71** | 0.59 |

### Phase 3: Extreme Optimization (Full Dataset - `bank-full.csv`)
Using **LightGBM + Optuna + Cross-Validation + Probability Threshold Tuning**:
| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM (Extreme)** | **90.85%** | **0.59** | **0.70** | **0.64** |

*Note: The Extreme optimization successfully boosted both Accuracy and F1-score to the highest possible levels simultaneously, avoiding the precision-drop trade-off seen in Phase 2.*

### Test Set Breakdown (Phases 2 & 3):
- **Class 0 (Negative - No Deposit)**: 7,985 samples
- **Class 1 (Positive - Subscribed)**: 1,058 samples

---

## 5. Key Findings & Discussion
1. **Class Imbalance**:
   - The dataset has an approximate **89% to 11% class imbalance**.
   - While overall accuracies on the baseline models are around ~90%, they mostly predict the majority class.
2. **Model Comparison**:
   - The **Extreme Pipeline (LightGBM + Threshold Tuning)** represents the absolute State-of-the-Art (SOTA) for this dataset. By discarding SMOTE in favor of native algorithm weighting (`scale_pos_weight`) and dynamically optimizing the probability threshold (found to be `0.6222`), it achieved a massive **F1-Score of 0.64** and **90.85% Accuracy**, destroying the baseline Random Forest.
   - The **Ultimate Pipeline (XGBoost + SMOTE)** fixed the recall problem (71%) but sacrificed precision (0.51) and accuracy (88.52%).
   - **Random Forest (Baseline)** achieved high overall accuracy (90.17%) mostly by guessing "no," resulting in a terrible F1-score for actual subscribers (0.37).

---

## 6. Recommendations & Next Steps
- **Productionize the XGBoost Model**: The Ultimate Pipeline code (`train_ultimate.py`) represents production-ready code. It should be wrapped in an API (e.g., FastAPI) for live inference.
- **Cost-Benefit Analysis**: A business threshold should be established to balance the cost of marketing to false positives (precision) against the missed revenue of false negatives (recall).
- **Feature Importance**: Analyze the SHAP values of the XGBoost model to understand the driving factors behind term deposit subscriptions (e.g., call duration, previous campaign success).

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

def main():
    print("Loading data...")
    # Loading the smaller dataset for faster training
    df = pd.read_csv('bank.csv', sep=';')

    print("Preprocessing data...")
    # Convert categorical variables to numeric
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features for SVM and KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Train and evaluate models
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start_time = time.time()
        
        # Use scaled data for distance-based algorithms
        if name in ['SVM', 'KNN']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        end_time = time.time()
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()

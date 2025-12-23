import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 1. Ambil Argumen dari MLProject (Dataset, Estimators, Depth)
    # sys.argv[3] adalah parameter dataset
    file_path = sys.argv[3] if len(sys.argv) > 3 else "heart_preprocessed.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} gak ketemu!")
        sys.exit(1)

    data = pd.read_csv(file_path)

    # 2. Split Data (Target: HeartDisease sesuai dataset lu)
    X = data.drop("HeartDisease", axis=1)
    y = data["HeartDisease"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )

    # 3. Ambil Parameter Model
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # 4. Training & Logging ke MLflow
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        # Log Model secara manual (Syarat Basic/Skilled)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train[0:5]
        )

        # Log Metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        
        print(f"✅ Berhasil! Akurasi: {accuracy:.4f} dengan n_est: {n_estimators}")

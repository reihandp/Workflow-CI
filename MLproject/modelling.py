import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Koneksi ke Server
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Heart_Failure_Prediction_Reihan")

# 2. Load Data
data = pd.read_csv("heart_preprocessed.csv")
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# 3. AKTIFKAN AUTOLOG DI AWAL 
mlflow.sklearn.autolog()

# 4. Training Model
with mlflow.start_run(run_name="RandomForest_Basic_Run"):
    # Pakai parameter simpel sesuai modul
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42
    )
    
    # Latih model
    model.fit(X_train, y_train)
    
    # Ambil skor akurasi
    accuracy = model.score(X_test, y_test)
    print(f"✅ Akurasi: {accuracy:.2f}")



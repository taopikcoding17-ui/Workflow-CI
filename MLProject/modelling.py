import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Aktifkan autolog (syarat Basic)
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("Default")
mlflow.autolog()

TRAIN_PATH = "iris_train_preprocessed.csv"
TEST_PATH  = "iris_test_preprocessed.csv"

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    X_test  = test_df.drop("target", axis=1)
    y_test  = test_df["target"]

    with mlflow.start_run(run_name="rf_iris_basic"):
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        print("Accuracy:", acc)
        print("F1 Macro:", f1)

        # Log metrics manual (biar jelas)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # WAJIB supaya folder model muncul di Artifacts
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
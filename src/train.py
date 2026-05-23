import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    le_dict = {}
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod',
                      'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                      'DeviceProtection', 'TechSupport', 'StreamingTV',
                      'StreamingMovies']
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    feature_columns = [c for c in df.columns if c != 'Churn']

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    artifacts = {
        "scaler": scaler,
        "numeric_cols": numeric_cols,
        "feature_columns": feature_columns,
        "label_encoders": le_dict,
        "multi_cat_cols": multi_cat_cols,
        "binary_cols": binary_cols
    }

    return X_train, X_test, y_train, y_test, artifacts

def train_and_log(X_train, X_test, y_train, y_test, preprocess_artifacts):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Save preprocessing artifacts (as before)
            os.makedirs("preprocess", exist_ok=True)
            for name, obj in preprocess_artifacts.items():
                joblib.dump(obj, f"preprocess/{name}.pkl")
            mlflow.log_artifacts("preprocess", artifact_path="preprocess")

            # Save the model manually as a plain file and log it
            os.makedirs("model", exist_ok=True)
            joblib.dump(model, "model/model.pkl")
            mlflow.log_artifacts("model", artifact_path="model")

            print(f"{model_name}: accuracy={acc:.3f}, f1={f1:.3f}")

if __name__ == "__main__":
    data_path = "data/raw/telco.csv"
    X_train, X_test, y_train, y_test, artifacts = preprocess_data(data_path)
    mlflow.set_experiment("Churn_Prediction")
    train_and_log(X_train, X_test, y_train, y_test, artifacts)
    print("Training complete. Run 'mlflow ui' to view results.")
import pandas as pd
import joblib

from feature_engineering import add_features


input_path = "data\processed\eda_telco_churn.csv"
output_path = "data\processed\churn_scoring_dataset.csv"


df = pd.read_csv(input_path)

df = add_features(df)

X = df.drop(columns=["Churn"])

model = joblib.load("models\churn_model_log.pkl")

df["churn_score"] = model.predict_proba(X)[:, 1]

df["risk_segment"] = pd.qcut(
    df["churn_score"],
    q=4,
    labels=[
        "Low risk",
        "Medium risk",
        "High risk",
        "Very high risk"
    ]
)

df.to_csv(output_path, index=False)

print("Scoring terminé")
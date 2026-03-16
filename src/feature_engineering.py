import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    services = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["num_services"] = df[services].apply(lambda x: (x == "Yes").sum(), axis=1)
    df["is_new_client"] = (df["tenure"] <= 12).astype(int)
    df["high_charges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df


if __name__ == "__main__":
    input_path = "../data/processed/eda_telco_churn.csv"
    output_path = "../data/processed/feature_engineered_telco_churn.csv"

    df = pd.read_csv(input_path)
    df = add_features(df)
    df.to_csv(output_path, index=False)

    print(f"Dataset sauvegardé : {output_path}")
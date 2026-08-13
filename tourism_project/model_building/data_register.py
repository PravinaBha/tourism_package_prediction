
import pandas as pd
import os

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]

def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    # Drop an accidental unnamed index column if present (common with exported CSVs)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    print("Dataset registered successfully.")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True))

    return df

if __name__ == "__main__":
    register_dataset()

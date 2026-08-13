
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

def prepare_data(path: str = DATA_PATH):
    df = pd.read_csv(path)

    # Drop stray index column if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Drop columns not useful for modeling
    df = df.drop(columns=["CustomerID"])

    # Separate features and target
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"]

    # Stratified split to preserve the ~19%/81% class balance in both sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
    print(f"ytrain distribution:\n{ytrain.value_counts(normalize=True)}")
    print(f"ytest distribution:\n{ytest.value_counts(normalize=True)}")

    return Xtrain, Xtest, ytrain, ytest

if __name__ == "__main__":
    prepare_data()


import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
import xgboost as xgb

CATEGORICAL_COLS = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
NUMERIC_COLS = ['Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
                'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
                'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome']

MODEL_OUT_PATH = "tourism_project/deployment/best_model.joblib"

def train_and_log():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("tourism_wellness_package")

    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").values.ravel()
    ytest = pd.read_csv("ytest.csv").values.ravel()

    # Ratio of negative to positive class, so XGBoost weights minority class (purchasers) more heavily
    neg, pos = (ytrain == 0).sum(), (ytrain == 1).sum()
    scale_pos_weight = neg / pos

    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        (StandardScaler(), NUMERIC_COLS),
    )

    pipeline = make_pipeline(
        preprocessor,
        xgb.XGBClassifier(eval_metric="logloss", random_state=42, scale_pos_weight=scale_pos_weight)
    )

    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    with mlflow.start_run():
        grid_search = GridSearchCV(
            pipeline, param_grid, scoring="f1", cv=5, n_jobs=-1
        )
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        ypred = best_model.predict(Xtest)

        metrics = {
            "accuracy": accuracy_score(ytest, ypred),
            "precision": precision_score(ytest, ypred),
            "recall": recall_score(ytest, ypred),
            "f1_score": f1_score(ytest, ypred),
        }
        mlflow.log_metrics(metrics)

        print("Best Parameters:", grid_search.best_params_)
        print("scale_pos_weight used:", scale_pos_weight)
        print("\nTest Set Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification Report:\n", classification_report(ytest, ypred))

        mlflow.sklearn.log_model(best_model, "model")

        os.makedirs("tourism_project/deployment", exist_ok=True)
        joblib.dump(best_model, MODEL_OUT_PATH)
        print(f"\nBest model saved to {MODEL_OUT_PATH}")

    return best_model, metrics

if __name__ == "__main__":
    train_and_log()

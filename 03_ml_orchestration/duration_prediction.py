#!/usr/bin/env python
# coding: utf-8

# Week 3: ML pipelines
# Data source: the NYC taxi dataset

import pandas as pd
import pickle

from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow
import xgboost as xgb


TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("NYC-taxi-experiment")

models_dir = Path('models')
models_dir.mkdir(exist_ok=True)

# embed all the preprocessing in a function
def read_dataframe(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    # create duration (in minutes) feature
    df["duration"] = (
        (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"])
        .dt.total_seconds()
        .div(60.0)
    )
    # filter out outliers: trips should be between 1 and 60 minutes.
    df = df[(1 <= df.duration) & (df.duration <= 60.0)]
    categorical_cols = ["PULocationID", "DOLocationID"]
    # convert categorical columns to string data type
    df[categorical_cols] = df[categorical_cols].astype(str)
    df["PU_DO"] = df[categorical_cols[0]] + "_" + df[categorical_cols[1]]
    return df


def create_X(df, dv=None):
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train, X_valid, y_valid, dv):

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_valid, label=y_valid)

        best_params = {
            "learning_rate": 0.2455,
            "max_depth": 90,
            "min_child_weight": 5.1467,
            "reg_alpha": 0.3175,
            "reg_lambda": 0.2810,
        }

        mlflow.log_params(best_params)
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_valid, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open(models_dir / "preprocessor.pkl", "wb") as fw:
            pickle.dump(dv, fw)
        mlflow.log_artifact(models_dir / "preprocessor.pkl", artifact_path="preprocessor")
        mlflow.xgboost.log_model(
            booster,
            artifact_path="models_mlflow",
        )
        return run.info.run_id

def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    valid_year = year if month < 12 else year + 1
    valid_month = month + 1 if month < 12 else 1
    df_valid = read_dataframe(year=valid_year, month=valid_month)

    X_train, dv = create_X(df_train)
    X_valid, _ = create_X(df_valid)

    target = "duration"
    y_train = df_train[target].values
    y_valid = df_valid[target].values

    run_id = train_model(X_train, y_train, X_valid, y_valid, dv)
    print(f"MLflow rund id: {run_id}")
    return run_id


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model to predict taxi trip duration"
    )
    parser.add_argument(
        '--year', type=int, required=True, help="Year of the data to train on",
    )
    parser.add_argument(
        '--month', type=int, required=True, help="Month of the data to train on",
    )
    args = parser.parse_args()
    run_id = run(year=args.year, month=args.month)
    with open("run_id.txt", "w") as fw:
        fw.write(run_id)




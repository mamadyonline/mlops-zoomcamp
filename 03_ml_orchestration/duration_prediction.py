#!/usr/bin/env python
# coding: utf-8

# Week 3: ML pipelines
# Data source: the NYC taxi dataset

import pandas as pd
import pickle

from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

import mlflow


TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("NYC-taxi-experiment")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)


# embed all the preprocessing in a function
def read_dataframe(year, month):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    print(f"Number of records in original dataset = {df.shape[0]}")
    # create duration (in minutes) feature
    df["duration"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
        .dt.total_seconds()
        .div(60.0)
    )
    # filter out outliers: trips should be between 1 and 60 minutes.
    df = df[(1 <= df.duration) & (df.duration <= 60.0)]
    print(f"Number of records in filtered dataset = {df.shape[0]}")
    categorical_cols = ["PULocationID", "DOLocationID"]
    # convert categorical columns to string data type
    df[categorical_cols] = df[categorical_cols].astype(str)
    return df


def create_X(df, dv=None):
    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Intercept = {model.intercept_}")

    return model


def run(year, month):
    with mlflow.start_run():
        df_train = read_dataframe(year=year, month=month)

        valid_year = year if month < 12 else year + 1
        valid_month = month + 1 if month < 12 else 1
        df_valid = read_dataframe(year=valid_year, month=valid_month)

        X_train, dv = create_X(df_train)
        X_valid, _ = create_X(df_valid, dv)

        target = "duration"
        y_train = df_train[target].values
        y_valid = df_valid[target].values

        model = train_model(X_train, y_train)
        y_pred = model.predict(X_valid)
        rmse_valid = root_mean_squared_error(y_valid, y_pred)
        
        with open(models_dir / "preprocessor.pkl", "wb") as fw:
            pickle.dump(dv, fw)

        mlflow.log_metric("rmse", rmse_valid)
        mlflow.log_artifact(
            models_dir / "preprocessor.pkl", artifact_path="preprocessor"
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="models_mlflow",
        )
    return dv, model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model to predict taxi trip duration"
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year of the data to train on",
    )
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Month of the data to train on",
    )
    args = parser.parse_args()
    print(f"Month = {args.month} | Year = {args.year}")

    # run_id = run(year=args.year, month=args.month)
    dv, model = run(year=args.year, month=args.month)
    # with open("run_id.txt", "w") as fw:
    #     fw.write(run_id)

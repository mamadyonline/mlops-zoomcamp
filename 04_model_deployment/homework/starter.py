#!/usr/bin/env python
# coding: utf-8


from pathlib import Path

import sys

import pickle
import pandas as pd


def read_data(filename, categorical):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


if __name__ == "__main__":
    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    year = int(input("Year : ").strip())
    month = int(input("Month : ").strip())

    categorical = ["PULocationID", "DOLocationID"]
    data_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    df = read_data(data_url, categorical)

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print(f"Standard deviation of prediction durations : {y_pred.std()}")
    print(f"Mean of prediction durations : {y_pred.mean()}")

    # preparing the dataframe
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    output_path = Path("data")
    output_path.mkdir(parents=True, exist_ok=True)

    df_result = pd.DataFrame({"ride_id": df.ride_id.values, "predictions": y_pred})

    output_file = output_path / "results.parquet"
    df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)

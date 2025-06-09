from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow


# MLflow configuration
TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("NYC-taxi-experiment")

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Default arguments for the DAG
default_args = {
    'owner': 'mamady',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 1),  # Start in April
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'nyc_taxi_ml_pipeline',
    default_args=default_args,
    description='NYC Taxi Trip Duration ML Pipeline',
    schedule='@monthly',  # Run monthly
    catchup=False,
    tags=['ml', 'taxi', 'regression'],
)

def read_dataframe(year, month):
    """Read and preprocess taxi data for given year and month"""
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    print(f"Number of records in original dataset = {df.shape[0]}")
    
    # Create duration (in minutes) feature
    df["duration"] = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
        .dt.total_seconds()
        .div(60.0)
    )
    
    # Filter out outliers: trips should be between 1 and 60 minutes
    df = df[(1 <= df.duration) & (df.duration <= 60.0)]
    print(f"Number of records in filtered dataset = {df.shape[0]}")
    
    categorical_cols = ["PULocationID", "DOLocationID"]
    # Convert categorical columns to string data type
    df[categorical_cols] = df[categorical_cols].astype(str)
    return df

def create_X(df, dv=None):
    """Create feature matrix from dataframe"""
    categorical = ["PULocationID", "DOLocationID"]
    dicts = df[categorical].to_dict(orient="records")
    
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv

def train_model(X_train, y_train):
    """Train linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Intercept = {model.intercept_}")
    return model

def calculate_train_validation_dates(**context):
    """Calculate training and validation dates based on execution date"""
    execution_date = context['execution_date']
    
    # Training data: 2 months before execution month
    train_month = execution_date.month - 2
    train_year = execution_date.year
    
    if train_month <= 0:
        train_month += 12
        train_year -= 1
    
    # Validation data: 1 month before execution month
    valid_month = execution_date.month - 1
    valid_year = execution_date.year
    
    if valid_month <= 0:
        valid_month += 12
        valid_year -= 1
    
    # Push dates to XCom for other tasks
    context['task_instance'].xcom_push(key='train_year', value=train_year)
    context['task_instance'].xcom_push(key='train_month', value=train_month)
    context['task_instance'].xcom_push(key='valid_year', value=valid_year)
    context['task_instance'].xcom_push(key='valid_month', value=valid_month)
    
    print(f"Execution date: {execution_date}")
    print(f"Training data: {train_year}-{train_month:02d}")
    print(f"Validation data: {valid_year}-{valid_month:02d}")

def load_and_preprocess_data(**context):
    """Load and preprocess training and validation data"""
    # Get dates from XCom
    train_year = context['task_instance'].xcom_pull(key='train_year')
    train_month = context['task_instance'].xcom_pull(key='train_month')
    valid_year = context['task_instance'].xcom_pull(key='valid_year')
    valid_month = context['task_instance'].xcom_pull(key='valid_month')
    
    # Load data
    print("Loading training data...")
    df_train = read_dataframe(year=train_year, month=train_month)
    
    print("Loading validation data...")
    df_valid = read_dataframe(year=valid_year, month=valid_month)
    
    # Create features
    print("Creating training features...")
    X_train, dv = create_X(df_train)
    
    print("Creating validation features...")
    X_valid, _ = create_X(df_valid, dv)
    
    target = "duration"
    y_train = df_train[target].values
    y_valid = df_valid[target].values
    
    # Store data in XCom (note: for large datasets, consider using external storage)
    context['task_instance'].xcom_push(key='X_train', value=X_train)
    context['task_instance'].xcom_push(key='X_valid', value=X_valid)
    context['task_instance'].xcom_push(key='y_train', value=y_train)
    context['task_instance'].xcom_push(key='y_valid', value=y_valid)
    context['task_instance'].xcom_push(key='vectorizer', value=dv)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_valid.shape}")

def train_and_evaluate_model(**context):
    """Train model and evaluate performance"""
    # Get data from XCom
    X_train = context['task_instance'].xcom_pull(key='X_train')
    X_valid = context['task_instance'].xcom_pull(key='X_valid')
    y_train = context['task_instance'].xcom_pull(key='y_train')
    y_valid = context['task_instance'].xcom_pull(key='y_valid')
    dv = context['task_instance'].xcom_pull(key='vectorizer')
    
    with mlflow.start_run():
        # Train model
        print("Training model...")
        model = train_model(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = model.predict(X_valid)
        
        # Calculate RMSE
        rmse_valid = root_mean_squared_error(y_valid, y_pred)
        print(f"Validation RMSE: {rmse_valid}")
        
        # Save preprocessor
        with open(models_dir / "preprocessor.pkl", "wb") as fw:
            pickle.dump(dv, fw)
        
        # Log metrics and artifacts to MLflow
        mlflow.log_metric("rmse", rmse_valid)
        mlflow.log_artifact(
            models_dir / "preprocessor.pkl", 
            artifact_path="preprocessor"
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="models_mlflow",
        )
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='rmse', value=rmse_valid)
        context['task_instance'].xcom_push(key='model', value=model)
        
        return f"Model trained successfully with RMSE: {rmse_valid:.4f}"

# Define tasks
calculate_dates_task = PythonOperator(
    task_id='calculate_dates',
    python_callable=calculate_train_validation_dates,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_and_preprocess_data',
    python_callable=load_and_preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_and_evaluate_model',
    python_callable=train_and_evaluate_model,
    dag=dag,
)

# Define task dependencies
calculate_dates_task >> load_data_task >> train_model_task
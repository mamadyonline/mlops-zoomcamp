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

# Debug print to confirm DAG is being loaded
print(f"DAG 'nyc_taxi_ml_pipeline' created successfully at {datetime.now()}")

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

def train_model_func(X_train, y_train):
    """Train linear regression model"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Intercept = {model.intercept_}")
    return model

def calculate_train_validation_dates(**context):
    """Calculate training and validation dates based on execution date"""
    # Use logical_date for newer Airflow versions, fallback to execution_date
    execution_date = context.get('logical_date') or context.get('execution_date')
    
    # Training data: 2 months before execution month
    train_month = execution_date.month - 4
    train_year = execution_date.year
    
    if train_month <= 0:
        train_month += 12
        train_year -= 1
    
    # Validation data: 1 month before execution month
    valid_month = execution_date.month - 3
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
    print(f"Training date: {train_year}-{train_month:02d}")
    print(f"Validation date: {valid_year}-{valid_month:02d}")

    return {
        "train_year": train_year,
        "train_month": train_month,
        "valid_year": valid_year,
        "valid_month": valid_month
    }

def load_and_preprocess_data(**context):
    """Load and preprocess training and validation data"""
    # Alternative: Get dates from return value of previous task
    date_info = context['task_instance'].xcom_pull(task_ids='calculate_dates')
    
    if date_info:
        train_year = date_info['train_year']
        train_month = date_info['train_month']
        valid_year = date_info['valid_year']
        valid_month = date_info['valid_month']
    else:
        # Fallback: try individual keys
        train_year = context['task_instance'].xcom_pull(task_ids='calculate_dates', key='train_year')
        train_month = context['task_instance'].xcom_pull(task_ids='calculate_dates', key='train_month')
        valid_year = context['task_instance'].xcom_pull(task_ids='calculate_dates', key='valid_year')
        valid_month = context['task_instance'].xcom_pull(task_ids='calculate_dates', key='valid_month')
    
    # Debug: Print what we got from XCom
    print(f"Retrieved from XCom - Train: {train_year}-{train_month}, Valid: {valid_year}-{valid_month}")
    
    # Validate that we got the data
    if None in [train_year, train_month, valid_year, valid_month]:
        raise ValueError(f"Missing date values from XCom: train_year={train_year}, train_month={train_month}, valid_year={valid_year}, valid_month={valid_month}")
    # Load data
    print("Loading training data...")
    df_train = read_dataframe(year=train_year, month=train_month)
    
    print("Loading validation data...")
    df_valid = read_dataframe(year=valid_year, month=valid_month)
    
    # # Create features
    print("Creating training features...")
    X_train, dv = create_X(df_train)
    
    print("Creating validation features...")
    X_valid, _ = create_X(df_valid, dv)
    
    target = "duration"
    y_train = df_train[target].values
    y_valid = df_valid[target].values
    
    # Store data using pickle files instead of XCom for large arrays
    # Create temporary directory for data exchange
    temp_dir = Path("/tmp/airflow_ml_data")
    temp_dir.mkdir(exist_ok=True)
    
    # Save arrays to files
    with open(temp_dir / "X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(temp_dir / "X_valid.pkl", "wb") as f:
        pickle.dump(X_valid, f)
    with open(temp_dir / "y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open(temp_dir / "y_valid.pkl", "wb") as f:
        pickle.dump(y_valid, f)
    with open(temp_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(dv, f)
    
    # Store file paths in XCom instead of large arrays
    context['task_instance'].xcom_push(key='data_dir', value=str(temp_dir))
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_valid.shape}")

def train_model(**context):
    """Train the model using training data"""
    # Set up MLflow inside the task function
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("NYC-taxi-experiment")
    
    # Get data from files
    print("Loading training data from files...")
    data_dir = Path(context['task_instance'].xcom_pull(task_ids='load_and_preprocess_data', key='data_dir'))
    
    with open(data_dir / "X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(data_dir / "y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open(data_dir / "vectorizer.pkl", "rb") as f:
        dv = pickle.load(f)

    print(f"Training data shape: {X_train.shape}")
    
    with mlflow.start_run() as run:
        # Train model
        print("Training model...")
        model = train_model_func(X_train, y_train)
        
        # Save model and preprocessor to files
        model_path = data_dir / "trained_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save preprocessor to models directory for MLflow
        preprocessor_path = models_dir / "preprocessor.pkl"
        with open(preprocessor_path, "wb") as fw:
            pickle.dump(dv, fw)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(
            preprocessor_path, 
            artifact_path="preprocessor"
        )
        mlflow.sklearn.log_model(
            model,
            artifact_path="models_mlflow",
        )
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='model_path', value=str(model_path))
        context['task_instance'].xcom_push(key='mlflow_run_id', value=run.info.run_id)
        
        print(f"Model trained successfully and saved to {model_path}")
        return f"Model training completed. MLflow run ID: {run.info.run_id}"


def validate_model(**context):
    """Validate the trained model using validation data"""
    # Set up MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    
    # Get data and model paths
    data_dir = Path(context['task_instance'].xcom_pull(task_ids='load_and_preprocess_data', key='data_dir'))
    model_path = context['task_instance'].xcom_pull(task_ids='train_model', key='model_path')
    mlflow_run_id = context['task_instance'].xcom_pull(task_ids='train_model', key='mlflow_run_id')
    
    # Load validation data
    print("Loading validation data from files...")
    with open(data_dir / "X_valid.pkl", "rb") as f:
        X_valid = pickle.load(f)
    with open(data_dir / "y_valid.pkl", "rb") as f:
        y_valid = pickle.load(f)
    
    # Load trained model
    print(f"Loading trained model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print(f"Validation data shape: {X_valid.shape}")
    
    # Continue the MLflow run from training
    with mlflow.start_run(run_id=mlflow_run_id):
        # Make predictions
        print("Making predictions on validation data...")
        y_pred = model.predict(X_valid)
        
        # Calculate RMSE
        rmse_valid = root_mean_squared_error(y_valid, y_pred)
        print(f"Validation RMSE: {rmse_valid}")
        
        # Log validation metrics to MLflow
        mlflow.log_metric("validation_rmse", rmse_valid)
        
        # Store results in XCom
        context['task_instance'].xcom_push(key='rmse', value=rmse_valid)
        context['task_instance'].xcom_push(key='model_path', value=model_path)  # Pass model path forward
        
        return f"Model validation completed with RMSE: {rmse_valid:.4f}"

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
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

# Define task dependencies
calculate_dates_task >> load_data_task >> train_model_task >> validate_model_task
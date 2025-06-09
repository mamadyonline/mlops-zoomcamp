from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

def hello_world():
    print("Hello World from Airflow!")

default_args = {
    'owner': 'test',
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'test_dag',
    default_args=default_args,
    description='A simple test DAG',
    schedule=timedelta(days=1),
    catchup=False,
)

task1 = PythonOperator(
    task_id='hello_task',
    python_callable=hello_world,
    dag=dag,
)
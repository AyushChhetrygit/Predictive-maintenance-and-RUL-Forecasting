from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def load_csv():
    print("Loading CSV...")

def validate_data():
    print("Validating data...")

def feature_engineering():
    print("Feature engineering...")

def write_to_db():
    print("Writing to TimescaleDB...")

with DAG(
    dag_id='batch_ingestion_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False
) as dag:

    t1 = PythonOperator(task_id='load_csv', python_callable=load_csv)
    t2 = PythonOperator(task_id='validate_data', python_callable=validate_data)
    t3 = PythonOperator(task_id='feature_engineering', python_callable=feature_engineering)
    t4 = PythonOperator(task_id='write_to_db', python_callable=write_to_db)

    t1 >> t2 >> t3 >> t4
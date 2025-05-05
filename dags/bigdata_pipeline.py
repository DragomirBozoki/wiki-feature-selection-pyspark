from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'dragomir',
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    dag_id="bigdata_wiki_pipeline",
    default_args=default_args,
    description="BigData Pipeline - Wikipedia classification",
    start_date=datetime(2024, 5, 1),
    schedule_interval='@daily',
    catchup=False,
    tags=["bigdata", "wiki"]
) as dag:

    step_1_run_main = BashOperator(
        task_id='run_main_script',
        bash_command='cd /mnt/c/Users/bozok/OneDrive/Desktop/Erasmus/BigData && source venv-bigdata/bin/activate && python main.py'
    )

    step_1_run_main

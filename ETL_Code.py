import requests
import pandas as pd
import mysql.connector
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Configure data source
source_url = 'https://api.example.com/data'
source_params = {
    'start_date': '2022-01-01',
    'end_date': '2022-01-31'
}

# Configure database destination
database_config = {
    'host': 'localhost',
    'database': 'mydatabase',
    'user': 'myuser',
    'password': 'mypassword'
}
table_name = 'mytable'

def extract_data():
    logging.info('Extracting data from source...')
    try:
        response = requests.get(source_url, params=source_params)
        response.raise_for_status()
        data = response.json()
        logging.info(f'Retrieved {len(data)} records.')
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f'Error retrieving data from source: {e}')
        raise

def transform_data(data):
    logging.info('Transforming data...')
    try:
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        logging.error(f'Error transforming data: {e}')
        raise

def load_data(df):
    logging.info('Loading data into database...')
    try:
        conn = mysql.connector.connect(**database_config)
        cur = conn.cursor()
        cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                age INT
            )
        ''')
        for index, row in df.iterrows():
            cur.execute('''
                INSERT INTO mytable (name, age)
                VALUES (%s, %s)
            ''', (row['name'], row['age']))
        conn.commit()
        cur.close()
        conn.close()
        logging.info(f'Loaded {len(df)} records into {table_name} table.')
    except Exception as e:
        logging.error(f'Error loading data into database: {e}')
        raise

if __name__ == '__main__':
    try:
        start_time = datetime.now()
        logging.info('Starting data pipeline...')
        data = extract_data()
        df = transform_data(data)
        load_data(df)
        end_time = datetime.now()
        logging.info(f'Pipeline completed in {end_time - start_time}.')
    except Exception as e:
        logging.error(f'Error running data pipeline: {e}')
        
        
        
# _______________ Application _____________________
sample_data = [
    {"name": "John", "age": 25},
    {"name": "Jane", "age": 30},
    {"name": "Alice", "age": 28}
]

if __name__ == '__main__':
    try:
        start_time = datetime.now()
        logging.info('Starting data pipeline...')
        data = sample_data  # Replace with your sample data
        df = transform_data(data)
        load_data(df)
        end_time = datetime.now()
        logging.info(f'Pipeline completed in {end_time - start_time}.')
    except Exception as e:
        logging.error(f'Error running data pipeline: {e}')


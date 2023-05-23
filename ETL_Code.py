# This code sets up a data pipeline to extract data from a source API, transform it using pandas, and load it into a 
# MySQL database. The pipeline is executed when the script is run as the main module (__name__ == '__main__'). 

# Here's a summary of what the code does:

# It imports the necessary libraries: requests, pandas, mysql.connector, logging, and datetime.
# It configures logging to log information and errors to a file called pipeline.log.
# It defines the source URL and parameters for the data extraction.
# It configures the database connection details and table name for data loading.
# The extract_data() function sends a GET request to the source URL with the specified parameters, 
# retrieves the JSON response, and logs the number of records retrieved.
# The transform_data() function takes the extracted data, converts it into a pandas DataFrame, and returns it.
# The load_data() function establishes a connection to the MySQL database, creates a table (if it doesn't exist),
# and inserts the data from the DataFrame into the table row by row.
# The __name__ == '__main__' condition ensures that the data pipeline is executed only when the script is run directly,
# not when imported as a module.
# The main code block executes the data pipeline by calling the extract_data(), transform_data(),
# and load_data() functions sequentially, measuring the execution time, and logging the completion status.

# ________________ ETL _______________________

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


import pandas as pd
from sqlalchemy import create_engine


# Database connection parameters
db_name = 'speeches'
db_user = 'pompos02'  # Replace with your PostgreSQL username
db_password = 'mypassword123'  # Replace with your PostgreSQL password
db_host = 'localhost'
db_port = '5432'

# Connect to PostgreSQL
engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Load CSV data
csv_file_path = 'tell_all_FILLED_FINAL.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path)

# Optional: Convert date column to datetime if needed
df['sitting_date'] = pd.to_datetime(df['sitting_date'], errors='coerce')

# Write DataFrame to PostgreSQL
table_name = 'speeches'
df.to_sql(table_name, con=engine, if_exists='replace', index=False)

print(f"Data loaded successfully into {table_name} table.")






import pandas as pd
from sqlalchemy import create_engine, text
import csv

# Database connection parameters
db_name = 'speeches'
db_user = 'pompos02'  # Replace with your PostgreSQL username
db_password = 'mypassword123'  # Replace with your PostgreSQL password
db_host = 'localhost'
db_port = '5432'
csv.field_size_limit(10**6)

# Connect to PostgreSQL
engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
speeches_count = pd.read_sql_query("SELECT COUNT(*) FROM speeches", con=engine)
print("Number of rows in speeches table:", speeches_count.iloc[0, 0])

merged_speeches_count = pd.read_sql_query("SELECT COUNT(*) FROM merged_speeches", con=engine)
print("Number of rows in merged_speeches table:", merged_speeches_count.iloc[0, 0])

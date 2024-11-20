import pandas as pd # type: ignore
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

# Load CSV data
csv_file_path = 'tell_all_FILLED_FINAL.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path, engine='python', encoding='utf-8')  # Or use 'latin1', 'utf-16', etc., if applicable

# Assuming 'sitting_date' column is in 'DD/MM/YYYY' format
df['sitting_date'] = pd.to_datetime(df['sitting_date'], format='%d/%m/%Y', errors='coerce')
df['sitting_date'] = df['sitting_date'].dt.strftime('%Y-%m-%d')

# Add an 'id' column (auto-incrementing)
df.insert(0, 'id', range(1, len(df) + 1))

# Write DataFrame to PostgreSQL
table_name = 'speeches'
df.to_sql(table_name, con=engine, if_exists='replace', index=False)

print(f"Data loaded successfully into {table_name} table.")

# Identify rows with corrupted dates
invalid_dates = df[pd.to_datetime(df['sitting_date'], errors='coerce').isna()]
if not invalid_dates.empty:
    print("Rows with corrupted dates:")
    print(invalid_dates)
else:
    print("No corrupted dates found.")

#NOMIZW EN DULEFKEI, DIMIURGA TO merged_speeches POU TO pgADMIN
# Drop 'merged_speeches' table if it exists and recreate it
drop_table_sql = "DROP TABLE IF EXISTS merged_speeches;"
create_table_sql = """
CREATE TABLE merged_speeches AS
SELECT
    ROW_NUMBER() OVER (ORDER BY member_name, parliamentary_period, parliamentary_session, parliamentary_sitting, sitting_date) AS id,
    member_name,
    parliamentary_period,
    parliamentary_session,
    parliamentary_sitting,
    sitting_date,
    STRING_AGG(speech, ' ') AS merged_speech,
    COUNT(*) AS speech_count
FROM
    speeches
GROUP BY
    member_name,
    parliamentary_period,
    parliamentary_session,
    parliamentary_sitting,
    sitting_date;
"""

# Execute the SQL commands
with engine.connect() as connection:
    connection.execute(text(drop_table_sql))  # Drop existing table if it exists
    connection.execute(text(create_table_sql))  # Create and populate the new table

print("Table 'merged_speeches' re-created and populated successfully.")

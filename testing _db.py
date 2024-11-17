import psycopg2

# Database connection parameters
db_name = 'speeches'
db_user = 'pompos02'  # Replace with your PostgreSQL username
db_password = 'mypassword123'  # Replace with your PostgreSQL password
db_host = 'localhost'
db_port = '5432'
# Connect using psycopg2
conn = psycopg2.connect(
    dbname=db_name,
    user=db_user,
    password=db_password,
    host=db_host,
    port=db_port
)
cur = conn.cursor()
cur.execute("SELECT * FROM speeches LIMIT 5;")
rows = cur.fetchall()
for row in rows:
    print(row)
conn.close()

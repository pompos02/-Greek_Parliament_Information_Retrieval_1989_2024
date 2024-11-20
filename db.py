from sqlalchemy import create_engine


def get_db():
    db_name = 'speeches'
    db_user = 'pompos02'  # Replace with your PostgreSQL username
    db_password = 'mypassword123'  # Replace with your PostgreSQL password
    db_host = 'localhost'
    db_port = '5432'
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    return engine

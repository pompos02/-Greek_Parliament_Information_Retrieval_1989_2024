import os


class Config:
    db_name = os.getenv('SUPABASE_DB_NAME', 'speeches')
    db_user = os.getenv('SUPABASE_DB_USER', 'your_supabase_user')
    db_password = os.getenv('SUPABASE_DB_PASSWORD', 'your_supabase_password')
    db_host = os.getenv('SUPABASE_DB_HOST', 'your_supabase_host')
    db_port = os.getenv('SUPABASE_DB_PORT', '5432')

    SQLALCHEMY_DATABASE_URI = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

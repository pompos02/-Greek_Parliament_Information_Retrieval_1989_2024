from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()


def get_db():
    db_name = os.getenv('SUPABASE_DB_NAME', 'speeches')
    db_user = os.getenv('SUPABASE_DB_USER', 'your_supabase_user')
    db_password = os.getenv('SUPABASE_DB_PASSWORD', 'your_supabase_password')
    db_host = os.getenv('SUPABASE_DB_HOST', 'your_supabase_host')
    db_port = os.getenv('SUPABASE_DB_PORT', '5432')

    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    return engine

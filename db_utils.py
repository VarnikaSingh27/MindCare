# db_utils.py
from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
_engine = create_engine(DATABASE_URL, future=True)

def read_table_as_df(table_name, limit=None):
    q = f"SELECT * FROM {table_name}"
    if limit:
        q += f" LIMIT {limit}"
    with _engine.connect() as conn:
        df = pd.read_sql(text(q), conn)
    return df
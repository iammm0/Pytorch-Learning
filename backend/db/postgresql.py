import psycopg2
from psycopg2.extras import RealDictCursor

def get_postgres_connection():
    conn = psycopg2.connect(
        dbname="growforever",
        user="postgres",
        password="iammm",
        host="localhost",
        port="5432"
    )
    return conn

def execute_query(query, params=None):
    with get_postgres_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            if query.strip().lower().startswith("select"):
                return cur.fetchall()

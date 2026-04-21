from dotenv import load_dotenv
import os
import psycopg2

def main():
    # Load environment variables from .env
    load_dotenv()

    # Read values
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    sslmode = os.getenv("POSTGRES_SSLMODE")

    print("Connecting to database...")

    # Connect to Postgres
    conn = psycopg2

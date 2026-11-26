import os
import ssl
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# Use Render's hosted database when DATABASE_URL is set.
database_url = os.getenv("DATABASE_URL")

if database_url:
    url = make_url(database_url)

    # Use the PostgreSQL driver already included in this project.
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+pg8000")
    elif url.drivername != "postgresql+pg8000":
        url = url.set(drivername="postgresql+pg8000")

else:
    # Local development database.
    url = URL.create(
        drivername="postgresql+pg8000",
        username="postgres",
        password=os.getenv("DB_PASSWORD"),
        host="localhost",
        port=5432,
        database="forcora_db"
    )

if database_url:
    query_params = dict(url.query)
    query_params.pop("sslmode", None)
    query_params.pop("channel_binding", None)

    url = url.set(query=query_params)

    engine = create_engine(
        url,
        connect_args={"ssl_context": ssl.create_default_context()}
    )
else:
    engine = create_engine(url)
    

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()
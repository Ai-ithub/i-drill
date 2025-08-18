from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import config

DATABASE_URL = (
    f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}"
    f"@{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
)

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_postgres_connection():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        print("Postgres connection failed:", e)
        return False

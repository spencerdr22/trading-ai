from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import load_config

cfg = load_config()
db_cfg = cfg["db"]

DATABASE_URL = f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['db']}"

# 🔍 Debug print - remove later
print("DATABASE_URL:", DATABASE_URL)

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session():
    return SessionLocal()

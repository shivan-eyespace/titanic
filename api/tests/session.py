"""Setting up TestSession."""

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from src.db.db import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=True, echo_pool="debug")
connection = engine.connect()
connection.begin()
TestSessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=connection)
)

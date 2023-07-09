"""Setting up pytest."""

import factory.random
import pytest
from fastapi.testclient import TestClient
from pytest_factoryboy import register
from sqlalchemy.orm import Session
from src.db import get_db, models
from src.main import app

from .factories import PassengerFactory
from .session import TestSessionLocal, connection

register(PassengerFactory)
factory.random.reseed_random("titanic")


@pytest.fixture(scope="session")
def db():
    """Database."""
    db = TestSessionLocal()
    yield db
    db.rollback()
    TestSessionLocal.remove()
    connection.close()


@pytest.fixture(scope="session")
def mocked_passengers(db: Session) -> list[models.Passenger]:
    """Mock passengers and added to database."""
    NUMBERS = 200
    db_passengers = [PassengerFactory() for _ in range(NUMBERS)]
    for passenger in db_passengers:
        db.add(passenger)
    db.commit()
    # db.refresh([db_passengers])
    return db_passengers


@pytest.fixture()
def mock_passenger() -> models.Passenger:
    """Provide a mock passenger for testing."""
    return PassengerFactory()


@pytest.fixture()
def mocked_passenger(db: Session) -> models.Passenger:
    """Already mocked passenger and added to database."""
    db_passenger = PassengerFactory()
    db.add(db_passenger)
    db.commit()
    db.refresh(db_passenger)
    return db_passenger


@pytest.fixture()
def client(db):
    """Client for running tests."""
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides = {}

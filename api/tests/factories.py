"""Factories for testing."""

import random

import factory
from src.db import models

from .session import TestSessionLocal


class PassengerFactory(factory.alchemy.SQLAlchemyModelFactory):
    """Passenger Factory."""

    class Meta:
        """Passenger Factory Meta."""

        model = models.Passenger
        sqlalchemy_session = TestSessionLocal

    pclass = factory.Iterator(
        [str(e.value) for e in models.Passenger.PclassEnum]
    )
    sex = factory.Iterator([e.value for e in models.Passenger.SexEnum])
    age = random.randint(1, 100)
    sibsp = random.randint(1, 5)
    parch = random.randint(1, 5)
    fare = random.random() * 100
    embarked = factory.Iterator(
        [e.value for e in models.Passenger.EmbarkedEnum]
    )

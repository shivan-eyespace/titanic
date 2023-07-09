"""Models."""

import enum

from sqlalchemy import Column, Enum, Float, Integer

from .db import Base


class Passenger(Base):
    """Passenger model."""

    PclassEnum = enum.Enum("Pclass", ["1", "2", "3"])

    class SexEnum(enum.Enum):
        """Enum for Sex."""

        m = "m"
        f = "f"

    class EmbarkedEnum(enum.Enum):
        """Enum for Embarked."""

        S = "S"
        C = "C"
        Q = "Q"

    __tablename__ = "passengers"
    id = Column("id", Integer, primary_key=True)
    pclass = Column("pclass", Enum(PclassEnum))
    sex = Column("sex", Enum(SexEnum))
    age = Column("age", Integer)
    sibsp = Column("sibsp", Integer)
    parch = Column("parch", Integer)
    fare = Column("fare", Float)
    embarked = Column("embarked", Enum(EmbarkedEnum))

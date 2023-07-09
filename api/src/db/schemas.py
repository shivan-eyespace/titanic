"""Schemas."""

from typing import Literal

from pydantic import BaseModel


class BasePassenger(BaseModel):
    """Passenger Model."""

    pclass: Literal["1", "2", "3"]
    sex: Literal["m", "f"]
    age: int
    sibsp: int
    parch: int
    fare: float
    embarked: Literal["S", "C", "Q"]


class CreatePassenger(BasePassenger):
    """Base Passenger Model."""

    pass


class Passenger(BasePassenger):
    """Passenger model."""

    id: int

    class Config:
        """Configuration for Passenger model."""

        orm_mode = True

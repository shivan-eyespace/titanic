"""CRUD operations."""

from sqlalchemy.orm import Session

from . import models, schemas


def count_passengers(db: Session) -> int:
    """Get passengers count."""
    count = db.query(models.Passenger).count()
    return count


def get_passengers(db: Session) -> list[models.Passenger]:
    """Get passengers."""
    db_passengers = db.query(models.Passenger).all()
    return db_passengers


def get_passenger_by_id(
    db: Session, passenger_id: int
) -> models.Passenger | None:
    """Get passenger by id."""
    db_passenger = (
        db.query(models.Passenger)
        .filter(models.Passenger.id == passenger_id)
        .one_or_none()
    )
    return db_passenger


def create_passenger(
    db: Session, passenger: schemas.CreatePassenger
) -> models.Passenger:
    """Create passenger."""
    db_passenger = models.Passenger(**passenger.dict())
    db.add(db_passenger)
    db.commit()
    db.refresh(db_passenger)
    return db_passenger


def update_passenger(
    db: Session, passenger_id: int, passenger: schemas.Passenger
):
    """Update passenger."""
    db_passenger = models.Passenger(passenger.dict())
    db.query(models.Passenger).where(
        models.Passenger.id == passenger_id
    ).update(db_passenger)
    db.commit()
    db.refresh(db_passenger)
    return db_passenger


def delete_passenger(db: Session, passenger_id: int) -> None:
    """Update passenger."""
    db.query(models.Passenger).filter(models.Passenger.id == id).delete()
    db.commit()

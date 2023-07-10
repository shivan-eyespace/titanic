"""V1 routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.db import crud, get_db, schemas
from src.nn import nn

router = APIRouter(prefix="/v1", tags=["v1"])

db = Depends(get_db)


@router.get("/passengers", response_model=list[schemas.Passenger])
async def read_passengers(cursor: str | None = None, db: Session = db):
    """Get all passengers."""
    passengers = crud.get_passengers(db=db)
    return passengers


@router.get("/passengers/{passenger_id}", response_model=schemas.Passenger)
async def read_passenger(passenger_id: int, db: Session = db):
    """Get a passenger based on id."""
    passenger = crud.get_passenger_by_id(db=db, passenger_id=passenger_id)
    if passenger is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Passenger not found"
        )
    return passenger


@router.post("/passengers", response_model=schemas.Passenger)
async def create_passenger(
    passenger: schemas.CreatePassenger, db: Session = db
):
    """Create a passenger."""
    passenger = crud.create_passenger(db=db, passenger=passenger)
    return passenger


@router.delete(
    "/passengers/{passenger_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_passenger(
    passenger_id: int, CreatePassenger, db: Session = db
):
    """Delete a passenger."""
    crud.delete_passenger(db=db, passenger_id=passenger_id)


@router.post("/predict")
async def predict(passenger: schemas.CreatePassenger):
    """Predict passenger survival."""
    predicition = nn.predict(passenger)
    return predicition

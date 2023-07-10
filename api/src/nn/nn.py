"""Scripts pertaining to the model."""

from tensorflow import keras
from pathlib import Path
from src.db import schemas

BASE_DIR = Path(__file__).parent.parent.parent

MODEL_DIR = BASE_DIR / "data" / "nn"

def predict(passengers: list[schemas.CreatePassenger]) -> list[float]:
    """Predict if passengers survived.

    Args:
    -----
    passengers: list[schemas.CreatePassenger]
        Passenger to predict survival

    Returns:
    --------
    tuple[bool, float]
        True if the passenger survived and the probability of survival.

    """
    # TODO: add preprocessing
    model = keras.models.load_model(MODEL_DIR)
    predicition = model.predict(passengers)
    return predicition

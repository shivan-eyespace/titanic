"""Tests for v1 endpoints."""

import pytest
from fastapi import status
from src.db import schemas


class TestV1:
    """Tests for V1 API."""

    url = "/v1"

    @pytest.mark.anyio
    def test_get_all_passengers(self, client, mocked_passengers):
        """Should give all passengers."""
        response = client.get(f"{self.url}/passengers")
        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == len(mocked_passengers)

    @pytest.mark.parametrize(
        "input_id,expected",
        [
            pytest.param(1, status.HTTP_200_OK, id="found"),
            pytest.param(-1, status.HTTP_404_NOT_FOUND, id="not found"),
        ],
    )
    @pytest.mark.anyio
    def test_get_passenger_by_id(
        self, client, mocked_passengers, input_id, expected
    ):
        """Should get a passenger by id.

        or display not found if id does not exist.
        """
        response = client.get(f"{self.url}/passengers/{input_id}")
        assert response.status_code == expected

    @pytest.mark.anyio
    def test_create_passenger(self, client, mock_passenger):
        """Should create a passenger."""
        passenger = schemas.CreatePassenger(**mock_passenger.__dict__).json()
        response = client.post(f"{self.url}/passengers", data=passenger)
        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.anyio
    def test_update_passenger(self, client, mock_passenger):
        """Should update a passenger."""
        pass

    @pytest.mark.anyio
    def test_delete_passenger(self, client, mocked_passenger):
        """Should delete a passenger."""
        response = client.delete(
            f"{self.url}/passengers/{mocked_passenger.id}"
        )
        assert response.status_code == status.HTTP_204_NO_CONTENT

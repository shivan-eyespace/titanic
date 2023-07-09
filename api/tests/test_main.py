"""Tests."""

import pytest
from fastapi import status


class TestDefault:
    """Tests for default."""

    url = ""

    @pytest.mark.anyio
    def test_root(self, client):
        """Should give a 200 response."""
        response = client.get(f"{self.url}/")
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.anyio
    def test_healthz(self, client):
        """Should give a 200 response."""
        response = client.get(f"{self.url}/healthz")
        assert response.status_code == status.HTTP_200_OK

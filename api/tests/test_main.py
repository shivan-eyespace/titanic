"""Tests."""

import pytest
from fastapi import status


class TestMain:
    """Tests for root."""

    url = ""

    @pytest.mark.anyio
    def test_root(self, client):
        """Testing the root."""
        response = client.get(f"{self.url}")
        assert response.status_code == status.HTTP_200_OK

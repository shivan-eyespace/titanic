"""API Service."""

from fastapi import FastAPI
from src import __version__
from src.routers import v1

description = """
Titanic dataset analysis.
"""

app = FastAPI(title="titanic")


@app.get("/")
async def root():
    """Root."""
    return {"message": "Welcome!"}


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"version": __version__}


# routes
app.include_router(v1.router)

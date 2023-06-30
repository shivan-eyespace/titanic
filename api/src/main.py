"""API Service."""

from fastapi import FastAPI

from . import __version__

app = FastAPI()


@app.get("/")
async def root():
    """Root."""
    return {"message": "Welcome!"}


@app.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"version": __version__}

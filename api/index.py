"""Vercel entrypoint for the FastAPI backend."""

from api.main import app

# Vercel function settings (can be overridden in vercel.json / dashboard).
config = {"maxDuration": 300}


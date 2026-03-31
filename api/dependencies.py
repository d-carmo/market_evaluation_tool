"""FastAPI dependency injectors — DB session, config, and API token auth."""
from __future__ import annotations

import os
from typing import Generator

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from config import AppConfig, load_config
from storage.session import get_session

# Singleton config — loaded once at startup
_config: AppConfig | None = None

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def verify_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Validate the X-API-Key header against the API_TOKEN env variable.

    If API_TOKEN is not set the service runs in open/dev mode and all requests
    are allowed (with a warning logged at startup).
    """
    expected = os.getenv("API_TOKEN") or ""
    if not expected:
        # Dev mode — no token configured
        return
    if api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def get_db() -> Generator[Session, None, None]:
    yield from get_session()

"""API server entry point: python serve.py  (or  uvicorn serve:app)"""
import logging

import uvicorn

from api.app import create_app
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

app = create_app()

if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        "serve:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
        log_level="info",
    )

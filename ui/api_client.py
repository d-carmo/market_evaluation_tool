"""Synchronous HTTP client for the Market Evaluation API."""
from __future__ import annotations

import os
from typing import Any

import httpx

_DEFAULT_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 60.0


class APIClient:
    def __init__(self, base_url: str = _DEFAULT_BASE, api_token: str = "") -> None:
        self.base = base_url.rstrip("/")
        self._headers = {"X-API-Key": api_token} if api_token else {}

    def _get(self, path: str, **params) -> Any:
        with httpx.Client(timeout=_TIMEOUT, verify=True) as client:
            r = client.get(f"{self.base}{path}", params=params, headers=self._headers)
            r.raise_for_status()
            return r.json()

    def _post(self, path: str, json: dict | None = None, **params) -> Any:
        with httpx.Client(timeout=_TIMEOUT, verify=True) as client:
            r = client.post(f"{self.base}{path}", json=json, params=params,
                            headers=self._headers)
            r.raise_for_status()
            return r.json()

    # --- Tickers ---

    def get_tickers(self) -> list[dict]:
        return self._get("/tickers/")

    def add_ticker(self, symbol: str, market_type: str | None = None) -> dict:
        return self._post("/tickers/", json={"symbol": symbol, "market_type": market_type})

    def remove_ticker(self, symbol: str) -> None:
        with httpx.Client(timeout=_TIMEOUT, verify=True) as client:
            r = client.delete(f"{self.base}/tickers/{symbol}", headers=self._headers)
            r.raise_for_status()

    # --- Analysis ---

    def analyze(self, ticker: str) -> dict:
        return self._post("/analyze/", json={"ticker": ticker})

    def get_latest(self, ticker: str) -> dict:
        return self._get(f"/analyze/{ticker}/latest")

    def get_chart_data(self, ticker: str) -> dict:
        return self._get(f"/analyze/{ticker}/chart")

    # --- Predictions ---

    def list_predictions(self) -> list[dict]:
        return self._get("/predictions/")

    def get_history(self, ticker: str, limit: int = 50) -> list[dict]:
        return self._get(f"/predictions/{ticker}/history", limit=limit)

    # --- Accuracy ---

    def accuracy_summary(self, ticker: str) -> list[dict]:
        return self._get(f"/accuracy/{ticker}/summary")

    def accuracy_records(self, ticker: str, horizon: int | None = None,
                         limit: int = 100) -> list[dict]:
        params: dict = {"limit": limit}
        if horizon is not None:
            params["horizon"] = horizon
        return self._get(f"/accuracy/{ticker}/records", **params)

    def trigger_evaluation(self) -> dict:
        return self._post("/accuracy/evaluate")

    # --- ML ---

    def train_all(self) -> dict:
        return self._post("/ml/train")

    def retrain_ticker(self, ticker: str, horizon: int | None = None) -> dict:
        params: dict = {}
        if horizon is not None:
            params["horizon"] = horizon
        return self._post(f"/ml/retrain/{ticker}", **params)

    # --- Health ---

    def health(self) -> dict:
        return self._get("/health")

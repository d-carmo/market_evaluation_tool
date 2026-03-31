"""Predictions routes: list stored predictions."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import Field
from sqlalchemy.orm import Session

from api.dependencies import get_db, verify_api_key
from api.schemas import StoredPredictionSchema, _validate_ticker

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/", response_model=list[StoredPredictionSchema],
            dependencies=[Depends(verify_api_key)])
def list_latest_predictions(db: Session = Depends(get_db)) -> list[StoredPredictionSchema]:
    """One latest prediction per ticker."""
    from storage.repository import PredictionRepository
    pred_repo = PredictionRepository(db)
    records = pred_repo.get_all_latest()
    return [StoredPredictionSchema.model_validate(r) for r in records]


@router.get("/{ticker}/history", response_model=list[StoredPredictionSchema],
            dependencies=[Depends(verify_api_key)])
def get_prediction_history(
    ticker: str,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[StoredPredictionSchema]:
    """Prediction history for a specific ticker."""
    ticker = _validate_ticker(ticker)
    if not 1 <= limit <= 500:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="limit must be between 1 and 500")
    from storage.repository import PredictionRepository
    pred_repo = PredictionRepository(db)
    records = pred_repo.get_history(ticker, limit=limit)
    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No predictions found for {ticker!r}")
    return [StoredPredictionSchema.model_validate(r) for r in records]

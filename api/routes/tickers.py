"""Ticker management routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from api.dependencies import get_db, verify_api_key
from api.schemas import _validate_ticker
from storage.orm_models import TickerRecord
from storage.repository import TickerRepository

router = APIRouter(prefix="/tickers", tags=["tickers"])


class TickerSchema(BaseModel):
    symbol: str
    market_type: str
    active: bool

    model_config = {"from_attributes": True}


class AddTickerRequest(BaseModel):
    symbol: str
    market_type: str | None = None  # auto-detected if omitted

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return _validate_ticker(v)


@router.get("/", response_model=list[TickerSchema], dependencies=[Depends(verify_api_key)])
def list_tickers(db: Session = Depends(get_db)) -> list[TickerSchema]:
    """Return all active tickers."""
    rows = db.query(TickerRecord).filter_by(active=True).all()
    return [TickerSchema(symbol=r.symbol, market_type=r.market_type, active=r.active) for r in rows]


@router.post("/", response_model=TickerSchema, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(verify_api_key)])
def add_ticker(req: AddTickerRequest, db: Session = Depends(get_db)) -> TickerSchema:
    """Add a new ticker (persisted to database)."""
    repo = TickerRepository(db)
    rec = repo.add(req.symbol, req.market_type)
    return TickerSchema(symbol=rec.symbol, market_type=rec.market_type, active=rec.active)


@router.delete("/{symbol}", status_code=status.HTTP_204_NO_CONTENT,
               dependencies=[Depends(verify_api_key)])
def remove_ticker(symbol: str, db: Session = Depends(get_db)) -> None:
    """Deactivate a ticker (soft delete)."""
    symbol = _validate_ticker(symbol)
    repo = TickerRepository(db)
    if not repo.remove(symbol):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Ticker {symbol!r} not found")

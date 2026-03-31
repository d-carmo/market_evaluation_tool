from pathlib import Path

from prediction.base import AbstractPredictor
from prediction.ml import MLPredictor
from prediction.statistical import StatisticalPredictor


def get_predictor(
    method: str = "ensemble",
    model_dir: Path | None = None,
    db=None,
) -> AbstractPredictor:
    """Return the appropriate predictor instance.

    Args:
        method:    "statistical" | "ml" | "lstm" | "ensemble"
        model_dir: Directory where model files are stored.
        db:        Optional SQLAlchemy Session (used by ensemble for blend weights).

    Raises:
        ValueError: for unknown method names.
    """
    if method == "statistical":
        return StatisticalPredictor()
    if method == "ml":
        return MLPredictor(model_dir=model_dir)
    if method == "lstm":
        from prediction.lstm_predictor import LSTMPredictor
        return LSTMPredictor(model_dir=model_dir)
    if method == "ensemble":
        from prediction.ensemble_predictor import EnsemblePredictor
        pred = EnsemblePredictor(model_dir=model_dir, include_rf=False)
        if db is not None:
            pred.set_db_session(db)
        return pred
    if method == "rf":
        from prediction.rf_predictor import RFPredictor
        return RFPredictor(model_dir=model_dir)
    if method == "rf_ensemble":
        from prediction.ensemble_predictor import EnsemblePredictor
        pred = EnsemblePredictor(model_dir=model_dir, include_rf=True)
        if db is not None:
            pred.set_db_session(db)
        return pred
    raise ValueError(
        f"Unknown predictor method: {method!r}. "
        "Choose 'statistical', 'ml', 'lstm', 'ensemble', 'rf', or 'rf_ensemble'."
    )

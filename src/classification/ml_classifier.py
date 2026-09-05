"""
ML Fire Classifier  (Stage 3)
Random Forest + XGBoost ensemble trained on rule-based labels + feature matrix.
Outputs per-class probabilities alongside the final classification label.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from src.preprocessing.feature_engineering import ML_FEATURES

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.info("XGBoost not installed — using RandomForest only.")

CLASS_ORDER = [
    "INDUSTRIAL_FIRE",
    "PERSISTENT_THERMAL",
    "WILDFIRE",
    "AGRICULTURAL_BURN",
    "UNKNOWN",
    "FALSE_POSITIVE",
]


class FireMLClassifier:
    """
    Ensemble fire classifier combining Random Forest (always) and
    XGBoost (when available). Trained on rule-based labels.

    Usage
    -----
    clf = FireMLClassifier(config)
    clf.train(feature_df, label_series)
    clf.predict(feature_df)   →  adds ml_class, ml_confidence, ml_proba_* columns
    clf.save() / clf.load()
    """

    def __init__(self, config: dict):
        self.cfg       = config
        self.model_path = Path(
            config.get("paths", {}).get("model_path", "data/model/fire_classifier.pkl")
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder   = LabelEncoder()
        self.encoder.classes_ = np.array(CLASS_ORDER)
        self.model: Optional[object] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the ensemble on the given feature matrix and labels."""
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.encoder = le

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        if _XGB_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )
            self.model = VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb)],
                voting="soft",
                weights=[1, 1],
            )
            logger.info("Training RF + XGBoost ensemble …")
        else:
            self.model = rf
            logger.info("Training Random Forest classifier …")

        X_clean = X[ML_FEATURES].fillna(0)
        self.model.fit(X_clean, y_enc)
        logger.info("ML classifier training complete.")

        # Inline evaluation
        preds = self.model.predict(X_clean)
        logger.info(
            "\n" + classification_report(
                y_enc, preds,
                target_names=le.classes_,
                zero_division=0,
            )
        )
        self.save()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with ML predictions and per-class probabilities.
        Columns: ml_class, ml_confidence, ml_proba_<ClassName>
        """
        if self.model is None:
            if self.model_path.exists():
                self.load()
            else:
                logger.warning("No trained ML model found — skipping ML stage.")
                result = pd.DataFrame(index=X.index)
                result["ml_class"]      = "UNKNOWN"
                result["ml_confidence"] = 0.5
                return result

        X_clean = X[ML_FEATURES].fillna(0)
        proba   = self.model.predict_proba(X_clean)   # (n_samples, n_classes)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels   = self.encoder.inverse_transform(pred_idx)
        pred_conf     = proba[np.arange(len(proba)), pred_idx]

        result = pd.DataFrame(index=X.index)
        result["ml_class"]      = pred_labels
        result["ml_confidence"] = pred_conf.round(4)

        # Per-class probabilities
        for i, cls in enumerate(self.encoder.classes_):
            col = f"ml_proba_{cls.lower()}"
            result[col] = proba[:, i].round(4)

        return result

    def save(self) -> None:
        payload = {"model": self.model, "encoder": self.encoder}
        joblib.dump(payload, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load(self) -> None:
        payload       = joblib.load(self.model_path)
        self.model    = payload["model"]
        self.encoder  = payload["encoder"]
        logger.info(f"Model loaded from {self.model_path}")

    def feature_importance(self) -> pd.Series | None:
        """Return feature importances if the model supports it."""
        if self.model is None:
            return None
        # VotingClassifier: extract RF importance
        estimator = self.model
        if hasattr(estimator, "estimators_"):
            rf_est = [e for n, e in estimator.estimators_ if n == "rf"]
            if rf_est:
                estimator = rf_est[0]
        if hasattr(estimator, "feature_importances_"):
            return pd.Series(estimator.feature_importances_, index=ML_FEATURES).sort_values(ascending=False)
        return None

# models/ensemble.py
import numpy as np
from utils.logger import setup_logger

logger = setup_logger(__name__)

class Ensemble:
    """Simple averaging ensemble of multiple models."""
    def __init__(self, models):
        self.models = models
        self.name = "ensemble"

    def predict(self, X):
        """Average predictions from all models."""
        preds = [model.predict(X) for model in self.models]
        return np.mean(preds, axis=0)

    def save(self, path):
        # Not implemented: we'd need to save all models separately
        logger.warning("Ensemble save not implemented; save individual models instead.")

    def load(self, path):
        # Not implemented
        logger.warning("Ensemble load not implemented; load individual models instead.")
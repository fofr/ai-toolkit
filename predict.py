# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
from typing import Optional

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        if weights is not None:
            self.weights = weights
            
    def predict(
        self,
        prompt: Path = Input(
            description="Please check the Train Tab at the top of the model to train a LoRA."
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if self.weights is None:
            return Path("output.zip")
        
        # Create an empty zip file called empty.zip
        os.system("touch empty.zip")
        return Path("empty.zip")

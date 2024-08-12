# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
from typing import Optional

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
            
    def predict(
        self,
        prompt: str = Input(
            description="Please check the Train Tab at the top of the model to train a LoRA."
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        os.system("touch empty.zip")
        return Path("empty.zip")

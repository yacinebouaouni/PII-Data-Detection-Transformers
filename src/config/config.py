import yaml
from typing import Any, Dict, List, Union

class Config:
    """
    Class to load and store configuration parameters from a YAML file.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the Config object.

        Args:
            path (str): Path to the YAML configuration file.
        """
        # Load the configuration from the YAML file
        self.config = self.load_config(path)
        
        # Set attributes based on the loaded configuration
        self.SEED: int = self.config.get("SEED")
        self.PATH_DATA: str = self.config.get("DATA").get("PATH_DATA")
        self.PATH_FOLDS: str = self.config.get("DATA").get("PATH_FOLDS")
        self.PATH_COMPETITION: str = self.config.get("DATA").get("PATH_COMPETITION")
        self.PATH_NICHOLAS: str = self.config.get("DATA").get("PATH_NICHOLAS")
        self.EXTRA_DATA: List[str] = self.config.get("DATA").get("EXTRA_DATA")
        self.FOLDS: int = self.config.get("DATA").get("FOLDS")
        self.TRAINING_MODEL_PATH: str = self.config.get("MODEL").get("TRAIN").get("TRAINING_MODEL_PATH")
        self.MODEL_SAVE: str = self.config.get("MODEL").get("TRAIN").get("MODEL_SAVE")
        self.TRAINING_MAX_LENGTH: int = self.config.get("MODEL").get("TRAIN").get("TRAINING_MAX_LENGTH")
        self.SAVE_MODELS: bool = self.config.get("MODEL").get("TRAIN").get("SAVE_MODELS")
        self.OUTPUT_DIR: str = self.config.get("MODEL").get("TRAIN").get("OUTPUT_DIR")
        self.BATCH: int = self.config.get("MODEL").get("TRAIN").get("BATCH")
        self.ACCUMULATION: int = self.config.get("MODEL").get("TRAIN").get("ACCUMULATION")
        self.WARMUP: int = self.config.get("MODEL").get("TRAIN").get("WARMUP")
        self.EPOCHS: float = self.config.get("MODEL").get("TRAIN").get("EPOCHS")
        self.LR: float = float(self.config.get("MODEL").get("TRAIN").get("LR"))
        self.STRIDE: int = self.config.get("MODEL").get("INFERENCE").get("STRIDE")
        self.INFERENCE_MAX_LENGTH: int = self.config.get("MODEL").get("INFERENCE").get("INFERENCE_MAX_LENGTH")
        self.THRESHOLD: float = self.config.get("MODEL").get("INFERENCE").get("THRESHOLD")

    def load_config(self, path: str) -> Dict[str, Any]:
        """
        Load the configuration from the specified YAML file.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            dict: Loaded configuration as a dictionary.
        """
        # Load configuration from the YAML file and return it as a dictionary
        with open(path, "r") as file:
            config: Dict[str, Any] = yaml.safe_load(file)
        return config
    
    

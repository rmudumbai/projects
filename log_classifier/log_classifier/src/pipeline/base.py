"""Base pipeline class with common functionality."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from loguru import logger
import json

class BasePipeline(ABC):
    def __init__(self, config_path: str = "dev/configs/config.yaml"):
        """Initialize the pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing configuration
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config['logging']
        logger.add(
            log_config['file'],
            level=log_config['level'],
            format=log_config['format']
        )
    
    def _ensure_dir(self, path: str):
        """Ensure directory exists.
        
        Args:
            path: Directory path
        """
        Path(path).mkdir(parents=True, exist_ok=True)
    
    def _save_metadata(self, data: Dict[str, Any], output_path: str):
        """Save metadata to JSON file.
        
        Args:
            data: Metadata to save
            output_path: Path to save metadata
        """
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_metadata(self, input_path: str) -> Dict[str, Any]:
        """Load metadata from JSON file.
        
        Args:
            input_path: Path to metadata file
            
        Returns:
            Dictionary containing metadata
        """
        with open(input_path, 'r') as f:
            return json.load(f)
    
    @abstractmethod
    def run(self, input_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the pipeline.
        
        Args:
            input_path: Optional input path
            output_path: Optional output path
            
        Returns:
            Dictionary containing pipeline results
        """
        pass 
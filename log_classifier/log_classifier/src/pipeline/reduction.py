"""Dimensionality reduction pipeline."""
from typing import Any, Dict, Optional
import numpy as np
import umap
from loguru import logger
from .base import BasePipeline

class ReductionPipeline(BasePipeline):
    def __init__(self, config_path: str = "dev/configs/config.yaml"):
        """Initialize the dimensionality reduction pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__(config_path)
        self.config = self.config['reduction']
        self._setup_reducer()
    
    def _setup_reducer(self):
        """Setup the UMAP reducer."""
        logger.info("Setting up UMAP reducer")
        self.reducer = umap.UMAP(
            n_components=self.config['n_components'],
            n_neighbors=self.config['n_neighbors'],
            min_dist=self.config['min_dist'],
            metric=self.config['metric']
        )
    
    def run(self, input_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the dimensionality reduction pipeline.
        
        Args:
            input_path: Path to input embeddings
            output_path: Optional output path
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info("Starting dimensionality reduction pipeline")
        
        # Load embeddings
        input_path = input_path or self.config['input_dir']
        embeddings = np.load(input_path)
        
        # Reduce dimensionality
        reduced_embeddings = self.reducer.fit_transform(embeddings)
        
        # Save reduced embeddings
        output_path = output_path or self.config['output_dir']
        self._ensure_dir(output_path)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_path}/reduced_embeddings_{timestamp}.npy"
        np.save(output_file, reduced_embeddings)
        
        # Save metadata
        metadata = {
            'n_samples': len(reduced_embeddings),
            'n_components': reduced_embeddings.shape[1],
            'input_file': input_path,
            'output_file': output_file,
            'timestamp': timestamp
        }
        metadata_path = f"{output_path}/metadata_{timestamp}.json"
        self._save_metadata(metadata, metadata_path)
        
        logger.info(f"Reduced {len(reduced_embeddings)} embeddings to {reduced_embeddings.shape[1]} dimensions")
        logger.info(f"Saved reduced embeddings to {output_file}")
        logger.info(f"Saved metadata to {metadata_path}")
        
        return metadata 
"""Log classification pipeline."""
from typing import Any, Dict, Optional
import numpy as np
import hdbscan
from loguru import logger
from .base import BasePipeline
import pandas as pd

class ClassificationPipeline(BasePipeline):
    def __init__(self, config_path: str = "dev/configs/config.yaml"):
        """Initialize the classification pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__(config_path)
        self.config = self.config['classification']
        self._setup_clusterer()
    
    def _setup_clusterer(self):
        """Setup the HDBSCAN clusterer."""
        logger.info("Setting up HDBSCAN clusterer")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config['min_cluster_size'],
            min_samples=self.config['min_samples'],
            metric=self.config['metric']
        )
    
    def run(self, input_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the classification pipeline.
        
        Args:
            input_path: Path to input embeddings
            output_path: Optional output path
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info("Starting classification pipeline")
        
        # Load embeddings
        input_path = input_path or self.config['input_dir']
        embeddings = np.load(input_path)
        
        # Cluster embeddings
        cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # Save cluster labels
        output_path = output_path or self.config['output_dir']
        self._ensure_dir(output_path)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_path}/cluster_labels_{timestamp}.npy"
        np.save(output_file, cluster_labels)
        
        # Calculate cluster statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        # Save metadata
        metadata = {
            'n_samples': len(cluster_labels),
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'input_file': input_path,
            'output_file': output_file,
            'timestamp': timestamp
        }
        metadata_path = f"{output_path}/metadata_{timestamp}.json"
        self._save_metadata(metadata, metadata_path)
        
        logger.info(f"Classified {len(cluster_labels)} embeddings into {n_clusters} clusters")
        logger.info(f"Found {n_noise} noise points")
        logger.info(f"Saved cluster labels to {output_file}")
        logger.info(f"Saved metadata to {metadata_path}")
        
        return metadata 
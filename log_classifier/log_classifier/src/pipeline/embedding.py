"""Log embedding pipeline."""
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from loguru import logger
from .base import BasePipeline

class LogEmbeddingPipeline(BasePipeline):
    def __init__(self, config_path: str = "dev/configs/config.yaml"):
        """Initialize the log embedding pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__(config_path)
        self.config = self.config['embedding']
        self._setup_model()
    
    def _setup_model(self):
        """Set up the model and tokenizer."""
        logger.info(f"Loading model: {self.config['model_name']}")
        
        # Load model without quantization
        self.model = AutoModel.from_pretrained(
            self.config['model_name'],
            device_map="auto"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name']
        )
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _read_log_file(self, file_path: str) -> str:
        """Read a log file and combine all lines into one story.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            Combined log story as a single string
        """
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        return ' '.join(lines)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Array containing the embedding
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors="pt"
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding
    
    def _process_file(self, file_path: str) -> tuple[np.ndarray, str]:
        """Process a single log file.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            Tuple of (embedding, filename)
        """
        logger.info(f"Processing file: {file_path}")
        
        # Read and combine log lines
        log_story = self._read_log_file(file_path)
        
        # Generate embedding
        embedding = self._generate_embedding(log_story)
        
        return embedding, Path(file_path).stem
    
    def run(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the log embedding pipeline.
        
        Args:
            input_path: Path to input file or directory
            output_path: Optional output path
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info(f"Starting log embedding pipeline with input: {input_path}")
        
        # Determine if input is file or directory
        input_path = Path(input_path)
        if input_path.is_file():
            files_to_process = [input_path]
        else:
            files_to_process = list(input_path.glob('*.txt'))
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process each file
        embeddings = []
        filenames = []
        
        for file_path in files_to_process:
            embedding, filename = self._process_file(str(file_path))
            embeddings.append(embedding)
            filenames.append(filename)
        
        # Combine all embeddings
        embeddings = np.vstack(embeddings)
        
        # Save embeddings
        output_path = output_path or self.config['output_dir']
        self._ensure_dir(output_path)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_path}/embeddings_{timestamp}.npy"
        np.save(output_file, embeddings)
        
        # Save metadata
        metadata = {
            'input_path': str(input_path),
            'output_file': output_file,
            'n_embeddings': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'timestamp': timestamp,
            'processed_files': filenames
        }
        metadata_path = f"{output_path}/metadata_{timestamp}.json"
        self._save_metadata(metadata, metadata_path)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        logger.info(f"Saved embeddings to {output_file}")
        logger.info(f"Saved metadata to {metadata_path}")
        
        return metadata 
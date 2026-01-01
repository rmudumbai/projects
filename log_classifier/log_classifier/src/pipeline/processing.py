"""Log processing pipeline."""
from typing import Any, Dict, Optional
import pandas as pd
import re
from loguru import logger
from .base import BasePipeline

class LogProcessingPipeline(BasePipeline):
    def __init__(self, config_path: str = "dev/configs/config.yaml"):
        """Initialize the log processing pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__(config_path)
        self.config = self.config['log_processing']
    
    def _read_log_file(self, file_path: str) -> list[str]:
        """Read log entries from a text file.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            List of log entries
        """
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _parse_log_entry(self, entry: str) -> Dict[str, Any]:
        """Parse a single log entry.
        
        Args:
            entry: Log entry string
            
        Returns:
            Dictionary containing parsed log components
        """
        # Parse level and message, ignoring timestamp
        pattern = r'\[.*?\] (\w+): (.*)'
        match = re.match(pattern, entry)
        
        if not match:
            logger.warning(f"Could not parse log entry: {entry}")
            return None
        
        level, message = match.groups()
        
        return {
            'level': level,
            'message': message,
            'sequence_id': None  # Will be filled in _process_logs
        }
    
    def _process_logs(self, log_entries: list[str]) -> pd.DataFrame:
        """Process log entries into a structured format.
        
        Args:
            log_entries: List of log entries
            
        Returns:
            DataFrame containing processed logs
        """
        processed_entries = []
        
        # Process entries and add sequence IDs
        for idx, entry in enumerate(log_entries):
            parsed = self._parse_log_entry(entry)
            if parsed:
                parsed['sequence_id'] = idx  # Add sequence ID to maintain order
                processed_entries.append(parsed)
        
        return pd.DataFrame(processed_entries)
    
    def run(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the log processing pipeline.
        
        Args:
            input_path: Path to input log file
            output_path: Optional output path
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info(f"Starting log processing pipeline with input: {input_path}")
        
        # Read log file
        log_entries = self._read_log_file(input_path)
        logger.info(f"Read {len(log_entries)} log entries")
        
        # Process logs
        processed_logs = self._process_logs(log_entries)
        logger.info(f"Processed {len(processed_logs)} log entries")
        
        # Save processed logs
        output_path = output_path or self.config['output_dir']
        self._ensure_dir(output_path)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_path}/processed_logs_{timestamp}.csv"
        processed_logs.to_csv(output_file, index=False)
        
        # Save metadata
        metadata = {
            'input_file': input_path,
            'output_file': output_file,
            'n_entries': len(processed_logs),
            'timestamp': timestamp
        }
        metadata_path = f"{output_path}/metadata_{timestamp}.json"
        self._save_metadata(metadata, metadata_path)
        
        logger.info(f"Saved processed logs to {output_file}")
        logger.info(f"Saved metadata to {metadata_path}")
        
        return metadata 
"""Log generation pipeline."""
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from .base import BasePipeline

class LogGenerationPipeline(BasePipeline):
    def __init__(self, config_path: str = "dev/configs/config.yaml"):
        """Initialize the log generation pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        super().__init__(config_path)
        self.config = self.config['log_generation']
    
    def _generate_logs(self, n_samples: int) -> list[str]:
        """Generate synthetic log data.
        
        Args:
            n_samples: Number of log samples to generate
            
        Returns:
            List of log entries in standard format
        """
        if n_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=24)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_samples)
        
        # Generate log levels with weighted distribution
        levels = np.random.choice(
            ['INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            size=n_samples,
            p=[0.6, 0.25, 0.1, 0.05]
        )
        
        # Generate log messages
        log_entries = []
        for timestamp, level in zip(timestamps, levels):
            if level == 'INFO':
                message = self._generate_info_message()
            elif level == 'WARNING':
                message = self._generate_warning_message()
            elif level == 'ERROR':
                message = self._generate_error_message()
            else:  # CRITICAL
                message = self._generate_critical_message()
            
            # Format log entry in standard format: [timestamp] LEVEL: message
            log_entry = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {level}: {message}"
            log_entries.append(log_entry)
        
        return log_entries
    
    def _generate_info_message(self) -> str:
        """Generate an INFO level log message."""
        templates = [
            "User {user_id} logged in successfully",
            "Request {request_id} completed in {duration}ms",
            "Cache hit for key {cache_key}",
            "Database connection pool size: {pool_size}",
            "Background task {task_id} started"
        ]
        return np.random.choice(templates).format(
            user_id=np.random.randint(1000, 9999),
            request_id=np.random.randint(10000, 99999),
            duration=np.random.randint(10, 1000),
            cache_key=f"key_{np.random.randint(1, 100)}",
            pool_size=np.random.randint(5, 20),
            task_id=np.random.randint(1000, 9999)
        )
    
    def _generate_warning_message(self) -> str:
        """Generate a WARNING level log message."""
        templates = [
            "High memory usage detected: {memory_usage}%",
            "Slow query detected: {query_time}ms for query {query_id}",
            "Cache miss for key {cache_key}",
            "Connection pool at {pool_usage}% capacity",
            "Retry attempt {attempt} for operation {operation_id}"
        ]
        return np.random.choice(templates).format(
            memory_usage=np.random.randint(70, 95),
            query_time=np.random.randint(1000, 5000),
            query_id=f"q_{np.random.randint(1000, 9999)}",
            cache_key=f"key_{np.random.randint(1, 100)}",
            pool_usage=np.random.randint(70, 95),
            attempt=np.random.randint(1, 5),
            operation_id=f"op_{np.random.randint(1000, 9999)}"
        )
    
    def _generate_error_message(self) -> str:
        """Generate an ERROR level log message."""
        templates = [
            "Failed to connect to database: {error_message}",
            "API request failed with status {status_code}: {error_message}",
            "File not found: {file_path}",
            "Invalid input data: {error_message}",
            "Task {task_id} failed after {attempts} attempts"
        ]
        return np.random.choice(templates).format(
            error_message=np.random.choice([
                "Connection timeout",
                "Authentication failed",
                "Permission denied",
                "Resource not found",
                "Invalid credentials"
            ]),
            status_code=np.random.choice([400, 401, 403, 404, 500, 503]),
            file_path=f"/path/to/file_{np.random.randint(1, 100)}.txt",
            task_id=f"task_{np.random.randint(1000, 9999)}",
            attempts=np.random.randint(1, 5)
        )
    
    def _generate_critical_message(self) -> str:
        """Generate a CRITICAL level log message."""
        templates = [
            "System shutdown initiated: {reason}",
            "Database connection lost: {error_message}",
            "Critical security breach detected: {details}",
            "Service {service_name} crashed: {error_message}",
            "Disk space critical: {free_space}MB remaining"
        ]
        return np.random.choice(templates).format(
            reason=np.random.choice([
                "Emergency maintenance",
                "Security incident",
                "Hardware failure",
                "Power outage"
            ]),
            error_message=np.random.choice([
                "Connection lost",
                "Authentication failed",
                "Permission denied",
                "Resource not found"
            ]),
            details=f"IP: 192.168.1.{np.random.randint(1, 255)}",
            service_name=np.random.choice([
                "auth-service",
                "payment-service",
                "user-service",
                "api-gateway"
            ]),
            free_space=np.random.randint(1, 100)
        )
    
    def run(self, input_path: Optional[str] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the log generation pipeline.
        
        Args:
            input_path: Not used in this pipeline
            output_path: Optional output path
            
        Returns:
            Dictionary containing pipeline results
        """
        logger.info("Starting log generation pipeline")
        
        # Generate logs
        log_entries = self._generate_logs(self.config['n_samples'])
        
        # Save logs
        output_path = output_path or self.config['output_dir']
        self._ensure_dir(output_path)
        
        timestamp = datetime.now().strftime(self.config['timestamp_format'])
        output_file = f"{output_path}/logs_{timestamp}.txt"
        
        # Write logs to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(log_entries))
        
        # Save metadata
        metadata = {
            'n_samples': len(log_entries),
            'output_file': output_file,
            'timestamp': timestamp
        }
        metadata_path = f"{output_path}/metadata_{timestamp}.json"
        self._save_metadata(metadata, metadata_path)
        
        logger.info(f"Generated {len(log_entries)} logs")
        logger.info(f"Saved logs to {output_file}")
        logger.info(f"Saved metadata to {metadata_path}")
        
        return metadata 
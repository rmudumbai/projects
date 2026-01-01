"""Tests for the log generation pipeline."""
import os
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from log_classifier.src.pipeline.generation import LogGenerationPipeline

@pytest.fixture
def temp_config():
    """Create a temporary config file."""
    config_content = """
    log_generation:
      n_samples: 100
      output_dir: "test_output"
      timestamp_format: "%Y%m%d_%H%M%S"
    logging:
      level: "INFO"
      format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
      file: "test.log"
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config_path = f.name
    
    yield temp_config_path
    
    # Cleanup
    os.unlink(temp_config_path)
    if os.path.exists("test_output"):
        for file in Path("test_output").glob("*"):
            file.unlink()
        Path("test_output").rmdir()
    if os.path.exists("test.log"):
        os.unlink("test.log")

def test_log_generation_pipeline_initialization(temp_config):
    """Test pipeline initialization."""
    pipeline = LogGenerationPipeline(temp_config)
    assert pipeline.config['n_samples'] == 100
    assert pipeline.config['output_dir'] == "test_output"
    assert pipeline.config['timestamp_format'] == "%Y%m%d_%H%M%S"

def test_generate_logs():
    """Test log generation functionality."""
    pipeline = LogGenerationPipeline()
    logs_df = pipeline._generate_logs(n_samples=10)
    
    # Check DataFrame structure
    assert isinstance(logs_df, pd.DataFrame)
    assert set(logs_df.columns) == {'timestamp', 'level', 'message'}
    assert len(logs_df) == 10
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(logs_df['timestamp'])
    assert pd.api.types.is_string_dtype(logs_df['level'])
    assert pd.api.types.is_string_dtype(logs_df['message'])
    
    # Check log levels
    valid_levels = {'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    assert set(logs_df['level'].unique()).issubset(valid_levels)

def test_pipeline_run(temp_config):
    """Test complete pipeline run."""
    pipeline = LogGenerationPipeline(temp_config)
    metadata = pipeline.run()
    
    # Check metadata structure
    assert isinstance(metadata, dict)
    assert 'n_samples' in metadata
    assert 'output_file' in metadata
    assert 'timestamp' in metadata
    
    # Check output file
    output_file = metadata['output_file']
    assert os.path.exists(output_file)
    
    # Check generated logs
    logs_df = pd.read_csv(output_file)
    assert len(logs_df) == 100
    assert set(logs_df.columns) == {'timestamp', 'level', 'message'}
    
    # Check metadata file
    timestamp = metadata['timestamp']
    metadata_file = f"test_output/metadata_{timestamp}.json"
    assert os.path.exists(metadata_file)

def test_custom_output_path(temp_config):
    """Test pipeline with custom output path."""
    custom_output = "custom_output"
    pipeline = LogGenerationPipeline(temp_config)
    metadata = pipeline.run(output_path=custom_output)
    
    assert os.path.exists(custom_output)
    assert os.path.exists(metadata['output_file'])
    assert custom_output in metadata['output_file']
    
    # Cleanup
    for file in Path(custom_output).glob("*"):
        file.unlink()
    Path(custom_output).rmdir()

def test_invalid_n_samples():
    """Test pipeline with invalid number of samples."""
    pipeline = LogGenerationPipeline()
    
    with pytest.raises(ValueError):
        pipeline._generate_logs(n_samples=0)
    
    with pytest.raises(ValueError):
        pipeline._generate_logs(n_samples=-1) 
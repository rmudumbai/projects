# Log Classification Pipeline

A modular Python pipeline for clustering and classifying error logs using advanced NLP and machine learning techniques.

## Project Overview

This project aims to automate the analysis of system logs by:
1. Processing raw log files into structured data
2. Generating semantic embeddings using a quantized Longformer model
3. Reducing dimensionality for efficient clustering
4. Classifying logs into meaningful clusters
5. Providing insights into system behavior and error patterns

## Technical Stack

### Core Technologies
- **Python 3.10+**: Main programming language
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for NLP models
- **Scikit-learn**: Machine learning utilities
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

### Key Components

#### 1. Log Processing
- Converts raw log files into structured format
- Handles both single files and directories
- Preserves log sequence and context
- Outputs processed logs in CSV format

#### 2. Embedding Generation
- Uses `allenai/longformer-base-4096` model
- Supports 8-bit quantization for memory efficiency
- Processes logs in batches
- Generates one embedding per log file
- Outputs embeddings in `.npy` format

#### 3. Dimensionality Reduction
- Uses UMAP (Uniform Manifold Approximation and Projection)
- Reduces high-dimensional embeddings to manageable size
- Preserves local and global structure
- Configurable parameters:
  - `n_components`: 10 (default)
  - `n_neighbors`: 15
  - `min_dist`: 0.1
  - `metric`: "cosine"

#### 4. Classification
- Uses HDBSCAN for density-based clustering
- Identifies natural clusters in the data
- Handles noise and outliers
- Configurable parameters:
  - `min_cluster_size`: 10
  - `min_samples`: 5
  - `metric`: "euclidean"

## Pipeline Architecture

```
Raw Logs (.txt) → Processing → Structured Data (.csv)
                              ↓
Embedding Generation → Vector Embeddings (.npy)
                              ↓
Dimensionality Reduction → Reduced Vectors
                              ↓
Classification → Clustered Results
```

### Detailed Flow

1. **Log Processing**
   - Input: Raw log files
   - Process: Parse and structure logs
   - Output: CSV with columns (sequence_id, level, message)

2. **Embedding Generation**
   - Input: Processed CSV
   - Process: 
     - Load quantized Longformer model
     - Generate embeddings for each log
     - Batch processing for efficiency
   - Output: NumPy array of embeddings

3. **Dimensionality Reduction**
   - Input: High-dimensional embeddings
   - Process:
     - UMAP transformation
     - Preserve semantic relationships
   - Output: Reduced-dimensional vectors

4. **Classification**
   - Input: Reduced vectors
   - Process:
     - HDBSCAN clustering
     - Identify patterns and anomalies
   - Output: Cluster assignments and metadata

## Configuration

The pipeline is configured through `dev/configs/config.yaml` with sections for:
- Log generation settings
- Processing parameters
- Model configuration
- Dimensionality reduction settings
- Classification parameters
- Logging configuration

## Usage

1. **Installation**
   ```bash
   pip install -e ".[dev]"
   ```

2. **Running Individual Stages**
   ```bash
   # Process logs
   python -m log_classifier.src.main process --input path/to/logs.txt

   # Generate embeddings
   python -m log_classifier.src.main embed --input path/to/processed_logs.csv

   # Reduce dimensions
   python -m log_classifier.src.main reduce --input path/to/embeddings.npy

   # Classify logs
   python -m log_classifier.src.main classify --input path/to/reduced_embeddings.npy
   ```

3. **Running Complete Pipeline**
   ```bash
   python -m log_classifier.src.main run-all
   ```

## Project Structure

```
log_classifier/
├── dev/
│   ├── configs/
│   │   └── config.yaml
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   ├── embeddings/
│   │   ├── reduced/
│   │   └── classified/
│   └── logs/
├── log_classifier/
│   └── src/
│       ├── pipeline/
│       │   ├── base.py
│       │   ├── generation.py
│       │   ├── processing.py
│       │   ├── embedding.py
│       │   ├── reduction.py
│       │   └── classification.py
│       └── main.py
└── tests/
    └── test_*.py
```

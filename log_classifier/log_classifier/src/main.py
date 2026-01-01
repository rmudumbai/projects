"""Main script to run the log classification pipeline."""
import click
from loguru import logger
from .pipeline import (
    LogGenerationPipeline,
    LogProcessingPipeline,
    LogEmbeddingPipeline,
    ReductionPipeline,
    ClassificationPipeline
)

@click.group()
def cli():
    """Log classification pipeline CLI."""
    pass

@cli.command()
@click.option('--config', default='dev/configs/config.yaml', help='Path to config file')
def generate(config):
    """Generate synthetic log data."""
    pipeline = LogGenerationPipeline(config)
    pipeline.run()

@cli.command()
@click.option('--config', default='dev/configs/config.yaml', help='Path to config file')
@click.option('--input', help='Path to input logs')
def process(config, input):
    """Process log data."""
    pipeline = LogProcessingPipeline(config)
    pipeline.run(input_path=input)

@cli.command()
@click.option('--config', default='dev/configs/config.yaml', help='Path to config file')
@click.option('--input', help='Path to input logs')
def embed(config, input):
    """Generate embeddings for log data."""
    pipeline = LogEmbeddingPipeline(config)
    pipeline.run(input_path=input)

@cli.command()
@click.option('--config', default='dev/configs/config.yaml', help='Path to config file')
@click.option('--input', help='Path to input embeddings')
def reduce(config, input):
    """Reduce dimensionality of embeddings."""
    pipeline = ReductionPipeline(config)
    pipeline.run(input_path=input)

@cli.command()
@click.option('--config', default='dev/configs/config.yaml', help='Path to config file')
@click.option('--input', help='Path to input embeddings')
def classify(config, input):
    """Classify log data."""
    pipeline = ClassificationPipeline(config)
    pipeline.run(input_path=input)

@cli.command()
@click.option('--config', default='dev/configs/config.yaml', help='Path to config file')
def run_all(config):
    """Run the entire pipeline."""
    # Generate logs
    logger.info("Generating logs")
    gen_pipeline = LogGenerationPipeline(config)
    gen_metadata = gen_pipeline.run()
    
    # Process logs
    logger.info("Processing logs")
    proc_pipeline = LogProcessingPipeline(config)
    proc_metadata = proc_pipeline.run(input_path=gen_metadata['output_file'])
    
    # Generate embeddings
    logger.info("Generating embeddings")
    emb_pipeline = LogEmbeddingPipeline(config)
    emb_metadata = emb_pipeline.run(input_path=proc_metadata['output_file'])
    
    # Reduce dimensionality
    logger.info("Reducing dimensionality")
    red_pipeline = ReductionPipeline(config)
    red_metadata = red_pipeline.run(input_path=emb_metadata['output_file'])
    
    # Classify logs
    logger.info("Classifying logs")
    cls_pipeline = ClassificationPipeline(config)
    cls_metadata = cls_pipeline.run(input_path=red_metadata['output_file'])
    
    logger.info("Pipeline completed successfully")

if __name__ == '__main__':
    cli() 
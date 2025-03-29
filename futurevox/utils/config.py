import yaml
import os
import logging


def load_config(config_path):
    """
    Load configuration from a YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        config: Dictionary containing configuration parameters
    """
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    _create_directories(config)
    
    return config


def _create_directories(config):
    """
    Create directories specified in the configuration
    
    Args:
        config: Configuration dictionary
    """
    # Create log directory if it doesn't exist
    if 'logging' in config and 'log_dir' in config['logging']:
        os.makedirs(config['logging']['log_dir'], exist_ok=True)
    
    # Create checkpoint directory if it doesn't exist
    if 'logging' in config and 'checkpoint_dir' in config['logging']:
        os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    
    # Create images directory if it doesn't exist
    if 'logging' in config and 'images_dir' in config['logging']:
        os.makedirs(config['logging']['images_dir'], exist_ok=True)


def parse_model_config(config, model_name):
    """
    Parse model configuration from the global configuration
    
    Args:
        config: Global configuration dictionary
        model_name: Name of the model to parse configuration for
        
    Returns:
        model_config: Dictionary containing model configuration
    """
    if 'model' not in config:
        raise ValueError("No model configuration found in config file")
    
    if model_name not in config['model']:
        raise ValueError(f"No configuration found for model: {model_name}")
    
    return config['model'][model_name]


def setup_logger(config):
    """
    Set up logger with configuration parameters
    
    Args:
        config: Configuration dictionary
        
    Returns:
        logger: Configured logger
    """
    # Create logger
    logger = logging.getLogger("futurevox")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # If log directory is specified, add file handler
    if 'logging' in config and 'log_dir' in config['logging']:
        log_dir = config['logging']['log_dir']
        log_file = os.path.join(log_dir, 'futurevox.log')
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    return logger
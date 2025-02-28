import os
import yaml
from typing import Dict, Any, Union
import logging

class ConfigLoader:
    """
    Advanced configuration loader with validation and environment variable support
    """
    
    @staticmethod
    def _convert_numeric_values(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively convert string representations of numeric values to appropriate types
        
        Args:
            config (Dict): Configuration dictionary
        
        Returns:
            Dict: Configuration with converted numeric values
        """
        def convert_value(value):
            if isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, str):
                try:
                    # Try converting to int
                    return int(value)
                except ValueError:
                    try:
                        # Try converting to float
                        return float(value)
                    except ValueError:
                        return value
            return value
        
        return convert_value(config)
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable expansion
        
        Args:
            config_path (str): Path to configuration file
        
        Returns:
            Dict of configuration parameters
        
        Raises:
            FileNotFoundError: If configuration file does not exist
            ValueError: If configuration is invalid
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Expand any environment variables in the path
        config_path = os.path.expandvars(config_path)
        
        # Check if file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
        
        # Convert string numeric values
        config = ConfigLoader._convert_numeric_values(config)
        
        # Validate critical sections
        ConfigLoader._validate_config(config)
        
        # Log successful configuration load
        logger.info(f"Loaded configuration from {config_path}")
        
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]):
        """
        Validate critical configuration parameters
        
        Args:
            config (Dict): Configuration dictionary
        
        Raises:
            ValueError: If critical configuration is missing or invalid
        """
        # Validation rules with more flexible checking
        def check_numeric(value: Union[int, float], name: str, min_val: float = 0):
            """Check if a value is a positive number"""
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number, got {type(value)}")
            if value <= min_val:
                raise ValueError(f"{name} must be greater than {min_val}")
        
        # Validate key sections and their parameters
        if 'paths' not in config:
            raise ValueError("Missing 'paths' section in configuration")
        
        # Training section validation
        if 'training' not in config:
            raise ValueError("Missing 'training' section in configuration")
        
        training_config = config['training']
        required_training_keys = [
            'batch_size', 
            'num_epochs', 
            'learning_rate', 
            'seed'
        ]
        
        for key in required_training_keys:
            if key not in training_config:
                raise ValueError(f"Missing required training parameter: {key}")
        
        # Numeric validations
        check_numeric(training_config['batch_size'], 'Batch size', 1)
        check_numeric(training_config['num_epochs'], 'Number of epochs', 1)
        check_numeric(training_config['learning_rate'], 'Learning rate', 0)
        
        # Model section validation
        if 'model' not in config:
            raise ValueError("Missing 'model' section in configuration")
        
        model_config = config['model']
        if 'feature_dim' not in model_config:
            raise ValueError("Missing 'feature_dim' in model configuration")
        
        check_numeric(model_config['feature_dim'], 'Feature dimension', 1)
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str):
        """
        Save configuration to a YAML file
        
        Args:
            config (Dict): Configuration dictionary
            output_path (str): Path to save configuration
        
        Raises:
            IOError: If unable to write the file
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save configuration
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {output_path}")
        
        except IOError as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries, with override_config taking precedence
        
        Args:
            base_config (Dict): Base configuration
            override_config (Dict): Configuration to override base settings
        
        Returns:
            Dict: Merged configuration
        """
        def deep_merge(dict1, dict2):
            """Recursively merge two dictionaries"""
            merged = dict1.copy()
            for key, value in dict2.items():
                if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key] = deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged
        
        return deep_merge(base_config, override_config)

def get_default_config() -> Dict[str, Any]:
    """
    Generate a default configuration with sensible defaults
    
    Returns:
        Dict: Default configuration dictionary
    """
    return {
        'paths': {
            'data_root': './data',
            'results_dir': './results',
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs'
        },
        'training': {
            'batch_size': 64,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'patience': 15,
            'seed': 42,
            'device': 'cuda',
            'max_grad_norm': 1.0,
            'label_smoothing': 0.1,
            'focal_gamma': 2.0,
            'warmup_epochs': 5
        },
        'model': {
            'feature_dim': 512,
            'num_heads': 4
        },
        'logging': {
            'log_every': 10
        },
        'wandb': {
            'enabled': False,
            'project': 'whale-mil'
        }
    }

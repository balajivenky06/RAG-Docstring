"""
Configuration loader utility for RAG system.
"""

import json
import os
from typing import Dict, Any
from .config import config, SystemConfig

def load_config_from_file(file_path: str) -> SystemConfig:
    """
    Load configuration from a JSON file.
    
    Args:
        file_path: Path to the configuration JSON file
        
    Returns:
        SystemConfig object with loaded configuration
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    
    # Update the global config with loaded values
    config.load_from_file(file_path)
    
    return config

def save_config_to_file(file_path: str, config_obj: SystemConfig = None) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        file_path: Path to save the configuration file
        config_obj: Configuration object to save (defaults to global config)
    """
    if config_obj is None:
        config_obj = config
    
    config_obj.save_to_file(file_path)

def create_config_from_template(template_path: str, output_path: str, 
                               pinecone_api_key: str = None) -> None:
    """
    Create a configuration file from template.
    
    Args:
        template_path: Path to the template file
        output_path: Path to save the new configuration
        pinecone_api_key: Pinecone API key to set
    """
    with open(template_path, 'r') as f:
        config_dict = json.load(f)
    
    # Set API key if provided
    if pinecone_api_key:
        config_dict['pinecone']['api_key'] = pinecone_api_key
    
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def validate_config_file(file_path: str) -> bool:
    """
    Validate a configuration file.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Check required fields
        required_fields = [
            'pinecone.api_key',
            'model.generator_model',
            'model.helper_model'
        ]
        
        for field in required_fields:
            keys = field.split('.')
            current = config_dict
            for key in keys:
                if key not in current:
                    print(f"Missing required field: {field}")
                    return False
                current = current[key]
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        print(f"Error validating configuration file: {e}")
        return False

def get_config_value(key_path: str, config_dict: Dict[str, Any] = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        key_path: Dot-separated path to the configuration value
        config_dict: Configuration dictionary (defaults to global config)
        
    Returns:
        Configuration value
    """
    if config_dict is None:
        config_dict = config.__dict__
    
    keys = key_path.split('.')
    current = config_dict
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    
    return current

def set_config_value(key_path: str, value: Any, config_dict: Dict[str, Any] = None) -> None:
    """
    Set a configuration value using dot notation.
    
    Args:
        key_path: Dot-separated path to the configuration value
        value: Value to set
        config_dict: Configuration dictionary (defaults to global config)
    """
    if config_dict is None:
        config_dict = config.__dict__
    
    keys = key_path.split('.')
    current = config_dict
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value

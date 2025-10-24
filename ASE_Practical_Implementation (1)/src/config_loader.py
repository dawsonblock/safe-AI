#!/usr/bin/env python3
"""
Configuration Loader for FDQC-Cockpit

Securely loads configuration from environment variables and .env files.
Provides centralized configuration management for all components.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Centralized production configuration"""
    
    # DeepSeek API
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    
    # BERT Configuration
    bert_model_name: str = "bert-base-uncased"
    bert_cache_dir: Optional[Path] = None
    bert_use_gpu: bool = True
    bert_batch_size: int = 32
    
    # Safety Configuration
    safety_tier: str = "GAMMA"
    require_human_approval: bool = True
    safe_mode: bool = True
    workspace_dim: int = 8
    entropy_threshold: float = 0.7
    collapse_threshold: float = 0.85
    
    # Persistence Configuration
    checkpoint_dir: Path = Path("./checkpoints")
    auto_save_interval_seconds: int = 300
    max_checkpoints: int = 10
    
    # Monitoring Configuration
    alert_risk_threshold: float = 0.8
    alert_error_rate_threshold: float = 0.1
    alert_memory_utilization_threshold: float = 0.9
    slack_webhook_url: Optional[str] = None
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "fdqc_cockpit.log"


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file
    
    Args:
        env_path: Path to .env file (defaults to .env in project root)
        
    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        # Look for .env in current directory and parent directories
        current = Path.cwd()
        for _ in range(5):  # Search up to 5 levels
            env_path = current / ".env"
            if env_path.exists():
                break
            current = current.parent
        else:
            logger.warning("No .env file found")
            return {}
    
    if not env_path.exists():
        logger.warning(f".env file not found at {env_path}")
        return {}
    
    env_vars = {}
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    env_vars[key] = value
        
        logger.info(f"Loaded {len(env_vars)} variables from {env_path}")
        
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
    
    return env_vars


def get_config(env_path: Optional[Path] = None) -> ProductionConfig:
    """
    Load production configuration from environment
    
    Priority:
    1. System environment variables
    2. .env file
    3. Default values
    
    Args:
        env_path: Optional path to .env file
        
    Returns:
        ProductionConfig instance
    """
    # Load .env file
    env_vars = load_env_file(env_path)
    
    # Merge with system environment (system takes priority)
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Helper to get env var with default
    def get_env(key: str, default: Any = None, required: bool = False) -> Any:
        value = os.getenv(key, default)
        
        if required and not value:
            raise ValueError(f"Required environment variable {key} not set")
        
        return value
    
    # Helper to parse boolean
    def parse_bool(value: str) -> bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    
    # Helper to parse int
    def parse_int(value: str, default: int) -> int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    # Helper to parse float
    def parse_float(value: str, default: float) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Build configuration
    config = ProductionConfig(
        # DeepSeek API
        deepseek_api_key=get_env('DEEPSEEK_API_KEY', required=True),
        deepseek_base_url=get_env('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1'),
        
        # BERT Configuration
        bert_model_name=get_env('BERT_MODEL_NAME', 'bert-base-uncased'),
        bert_cache_dir=Path(get_env('BERT_CACHE_DIR', './models/bert_cache')),
        bert_use_gpu=parse_bool(get_env('BERT_USE_GPU', 'true')),
        bert_batch_size=parse_int(get_env('BERT_BATCH_SIZE', '32'), 32),
        
        # Safety Configuration
        safety_tier=get_env('SAFETY_TIER', 'GAMMA'),
        require_human_approval=parse_bool(get_env('REQUIRE_HUMAN_APPROVAL', 'true')),
        safe_mode=parse_bool(get_env('SAFE_MODE', 'true')),
        workspace_dim=parse_int(get_env('WORKSPACE_DIM', '8'), 8),
        entropy_threshold=parse_float(get_env('ENTROPY_THRESHOLD', '0.7'), 0.7),
        collapse_threshold=parse_float(get_env('COLLAPSE_THRESHOLD', '0.85'), 0.85),
        
        # Persistence Configuration
        checkpoint_dir=Path(get_env('CHECKPOINT_DIR', './checkpoints')),
        auto_save_interval_seconds=parse_int(get_env('AUTO_SAVE_INTERVAL_SECONDS', '300'), 300),
        max_checkpoints=parse_int(get_env('MAX_CHECKPOINTS', '10'), 10),
        
        # Monitoring Configuration
        alert_risk_threshold=parse_float(get_env('ALERT_RISK_THRESHOLD', '0.8'), 0.8),
        alert_error_rate_threshold=parse_float(get_env('ALERT_ERROR_RATE_THRESHOLD', '0.1'), 0.1),
        alert_memory_utilization_threshold=parse_float(get_env('ALERT_MEMORY_UTILIZATION_THRESHOLD', '0.9'), 0.9),
        slack_webhook_url=get_env('SLACK_WEBHOOK_URL'),
        
        # Logging Configuration
        log_level=get_env('LOG_LEVEL', 'INFO'),
        log_file=get_env('LOG_FILE', 'fdqc_cockpit.log')
    )
    
    # Create directories if they don't exist
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if config.bert_cache_dir:
        config.bert_cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Configuration loaded successfully")
    logger.info(f"  DeepSeek API: {'*' * 20}{config.deepseek_api_key[-8:]}")
    logger.info(f"  BERT Model: {config.bert_model_name}")
    logger.info(f"  Safety Tier: {config.safety_tier}")
    logger.info(f"  Safe Mode: {config.safe_mode}")
    
    return config


def validate_config(config: ProductionConfig) -> bool:
    """
    Validate configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Validate API key format
    if not config.deepseek_api_key.startswith('sk-'):
        raise ValueError("Invalid DeepSeek API key format (should start with 'sk-')")
    
    # Validate safety tier
    valid_tiers = ['ALPHA', 'BETA', 'GAMMA', 'DELTA', 'EPSILON', 'OMEGA']
    if config.safety_tier not in valid_tiers:
        raise ValueError(f"Invalid safety tier: {config.safety_tier}. Must be one of {valid_tiers}")
    
    # Validate thresholds
    if not 0 <= config.entropy_threshold <= 1:
        raise ValueError("entropy_threshold must be between 0 and 1")
    
    if not 0 <= config.collapse_threshold <= 1:
        raise ValueError("collapse_threshold must be between 0 and 1")
    
    if not 0 <= config.alert_risk_threshold <= 1:
        raise ValueError("alert_risk_threshold must be between 0 and 1")
    
    # Validate directories
    if not config.checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory does not exist: {config.checkpoint_dir}")
    
    logger.info("✓ Configuration validation passed")
    return True


if __name__ == "__main__":
    # Test configuration loading
    print("Testing Configuration Loader...")
    
    try:
        config = get_config()
        validate_config(config)
        
        print("\n✓ Configuration loaded and validated successfully")
        print(f"\nConfiguration Summary:")
        print(f"  DeepSeek API Key: {'*' * 20}{config.deepseek_api_key[-8:]}")
        print(f"  BERT Model: {config.bert_model_name}")
        print(f"  BERT GPU: {config.bert_use_gpu}")
        print(f"  Safety Tier: {config.safety_tier}")
        print(f"  Safe Mode: {config.safe_mode}")
        print(f"  Checkpoint Dir: {config.checkpoint_dir}")
        print(f"  Log Level: {config.log_level}")
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nPlease check your .env file and ensure all required variables are set.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")

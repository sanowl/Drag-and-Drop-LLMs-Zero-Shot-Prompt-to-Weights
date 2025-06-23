"""
Logging configuration utilities.
"""

import logging
import sys
from typing import Optional
import os
from datetime import datetime


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration exactly as described in paper
    """
    if format_string is None:
        format_string = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set root logger level
    root_logger.setLevel(numeric_level)
    
    # Return main logger
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {level}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


def create_log_file(output_dir: str, prefix: str = "dnd_llm") -> str:
    """Create a timestamped log file path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{prefix}_{timestamp}.log"
    return os.path.join(output_dir, "logs", log_filename) 
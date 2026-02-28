import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def get_logger(name):
    """
    Get a configured logger instance.
    Args:
        name: The name of the logger (usually __name__)
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # If logger already has handlers, assume it's configured and return it
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 1. Stream Handler (Console)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # 2. File Handler (Rotating)
    # Log to a file, max 5MB, keep 3 backups
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "daisy.log"),
        maxBytes=5*1024*1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

import logging

def logger(file_name, level, message):
    """
    A custom logger function to log messages to both a file and the console.

    Args:
        file_name (str): The path to the log file where logs will be written.
        level (int): The logging level (e.g., logging.INFO, logging.ERROR, etc.).
        message (str): The message to log.

    Behavior:
        - Logs messages to the specified file and the console simultaneously.
        - Clears existing handlers to prevent duplicate logs.
        - Supports five logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        - Adds timestamps and log levels to the messages for readability.

    File Logging:
        - Logs all levels to the specified file using a formatter that includes timestamps and log levels.

    Console Logging:
        - Logs all levels to the terminal with the same format as the file.

    Usage Example:
        >>> from logging import INFO, ERROR
        >>> logger("application.log", logging.INFO, "This is an info message.")
        >>> logger("application.log", logging.ERROR, "This is an error message.")

    Raises:
        None: Handles incorrect logging levels by defaulting to `INFO` level.

    Notes:
        - Each logger is uniquely named based on the provided file name to avoid conflicts.
        - Existing handlers are cleared before adding new ones to prevent duplicate logs.
    """    
    # Use a unique logger name based on the file name
    logger_name = f"Nasir_{file_name}"
    logger1 = logging.getLogger(logger_name)
    logger1.setLevel(logging.DEBUG)  # Set the base level to DEBUG to capture all levels
    
    # Clear existing handlers to avoid duplication
    if logger1.hasHandlers():
        logger1.handlers.clear()
    
    # File handler for logging to a file
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)  # Log all levels to the file
    
    # Define a formatter and set it for the file handler
    file_formatter = logging.Formatter("[%(asctime)s] - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    
    # Add the file handler to the logger
    logger1.addHandler(file_handler)
    
    # Console handler for logging to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Log all levels to the console
    
    # Define a formatter and set it for the console handler
    console_formatter = logging.Formatter("[%(asctime)s] - %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    # Add the console handler to the logger
    logger1.addHandler(console_handler)
    
    # Log the message using the specified log level
    if level == logging.INFO:
        logger1.info(message)
    elif level == logging.ERROR:
        logger1.error(message)
    elif level == logging.WARNING:
        logger1.warning(message)
    elif level == logging.DEBUG:
        logger1.debug(message)
    elif level == logging.CRITICAL:
        logger1.critical(message)
    else:
        logger1.info(message)  # Default to INFO if an unknown level is provided
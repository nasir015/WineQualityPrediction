import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed information about an error, including the script name,
    line number, and the error message.

    Args:
        error (Exception): The exception object to extract details from.
        error_detail (sys): The sys module to access traceback details.

    Returns:
        str: A formatted error message with the script name, line number, and error description.

    Example:
        >>> try:
        >>>     1 / 0
        >>> except Exception as e:
        >>>     print(error_message_detail(e, sys))
        Error occurred in python script name [script_name.py] line number [line_no] error message[division by zero]
    """    
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    """
    A custom exception class for handling and displaying detailed error messages.

    Attributes:
        error_message (str): A detailed error message generated from the provided exception.

    Methods:
        __str__:
            Returns the detailed error message when the exception is converted to a string.
    """    
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with a detailed error message.

        Args:
            error_message (Exception): The original exception object.
            error_detail (sys): The sys module for accessing traceback details.
        """        
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        """
        Returns the detailed error message.

        Returns:
            str: The formatted error message.
        """        
        return self.error_message
class OutputFileNotFoundError(Exception):
    """Exception raised when the expected output file is not found."""
    def __init__(self, filepath: str = ""):
        message = f"Output file not found: {filepath}" if filepath else "Output file not found."
        super().__init__(message)

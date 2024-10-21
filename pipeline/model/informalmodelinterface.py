class InformalModelInterface:
    def splitTrainTestVal(self, X, y, trainSize: float, testSize: float):
        """Splits the data and target into a Train, Test, and Validation."""
        pass

    def extract_text(self, full_file_name: str):
        """Extract text from the currently loaded file."""
        pass

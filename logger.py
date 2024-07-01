import numpy as np
import pandas as pd
from typing import Dict

class PeptideLogger():
    def __init__(self, peptide_sequence: str, file_name_retention_times: float) -> None:
        self.peptide_sequence = peptide_sequence
        self.file_name_retention_time = {}
        self.transformed_retention_time = None

    def get_retention_times(self) -> np.ndarray:
        return np.array(list(self.file_name_retention_time.values()))
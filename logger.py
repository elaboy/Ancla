import numpy as np
import pandas as pd
from typing import Dict

class PeptideLogger():
    def __init__(self, peptide_sequence: str) -> None:
        self.peptide_sequence = peptide_sequence
        self.file_name_retention_time = {}
        self.transformed_retention_time = {}
        self.master_file_name_retention_time = {}

    def get_retention_times(self) -> np.ndarray:
        return np.array(list(self.file_name_retention_time.values())).reshape(-1)
    
    def get_transformed_retention_times(self) -> np.ndarray:
        return np.array(list(self.transformed_retention_time.values())).reshape(-1)

    def get_master_retention_times(self) -> np.ndarray:
        return np.array(list(self.master_file_name_retention_time.values())).reshape(-1)

    def update_file_name_retention_time(self, file_name: str, retention_time: float) -> None:
        self.file_name_retention_time[file_name] = retention_time

    def update_transformed_retention_times(self, file_name: str, transformed_retention_time: float) -> None:
        self.transformed_retention_time[file_name] = transformed_retention_time

    def update_master_file_name_retention_time(self, file_name: str, retention_time: float) -> None:
        self.master_file_name_retention_time[file_name] = retention_time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from tqdm import tqdm
from enum import Enum
from sklearn.base import clone #lets me clone the model parameters, good for tuning parameters and data preprocessing trials

class Calibrator():
    def __init__(self, path_list: list) -> None:
        self.path_list = path_list
        self.raw_file_names = {}
        self.dataframe_dictionary = {}
        self.calibration_data = pd.DataFrame()
        self.full_sequences = {}
        self.calibration_data['x'] = []
        self.calibration_data['y'] = []
        self.calibration_data['slope'] = []
        self.calibration_data['intercept'] = []
        self.calibration_data['r_value'] = []
        self.calibration_data['p_value'] = []
        self.calibration_data['std_err'] = []
        self.linear_regression_model = LinearRegression()

    def calibrate(self) -> pd.DataFrame:
        self.__read_data()
        self.__split_by_raw_file()
        self.__make_dataframes()
        self.__calibrate()
        return self.calibration_data

    def __calibrate(self) -> None:

        return None
    
    def __get_anchors(self, lead_df: pd.DataFrame, follower_df: pd.DataFrame) -> pd.DataFrame:
        # filter data
        filtered_lead_df = self.__filter_data(lead_df)
        filtered_follower_df = self.__filter_data(follower_df)
        # reduce variance
        filtered_lead_df = self.__reduce_variance(filtered_lead_df)
        filtered_follower_df = self.__reduce_variance(filtered_follower_df)
        # get anchors
        anchors = pd.merge(lead_df, follower_df, on='Full Sequence', how='inner')
        return anchors

    def __read_data(self) -> None:
        for path in self.path_list:
            self.df_list.append(pd.read_csv(path))
    

    def __split_by_raw_file(self) -> None:
        for i, df in enumerate(self.df_list):
            self.raw_file_names[i] = df['File Name'].unique()

    def __make_dataframes(self) -> None:
        for i, df in enumerate(self.df_list):
            for raw_file in self.raw_file_names[i]:
                self.dataframe_dictionary[raw_file] = df[df['File Name'] == raw_file]

    def __filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # fiter by q_value
        df = df[df['q_value'] < 0.01]
        # filter by PEP
        df = df[df['PEP'] < 0.5]
        # filter by ambiguity
        df = df[df['Ambiguity'] == "1"]
        #filter by "Decoy/Contaminant/Target"
        df = df[df['Decoy/Contaminant/Target'] == PsmType.TARGET.value]
        return df
    
    def __reduce_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        z_scores = np.abs(stats.zscore(df['Scan Retention Time']))
        df = df[(z_scores < 3) & (z_scores > -3)]
        return df
    
class PsmType(Enum):
    TARGET = "T"
    DECOY = "D"
    CONTAMINANT = "C"
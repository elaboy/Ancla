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
        self.master_file = None
        self.follower_files = []
        self.raw_file_names = {}
        self.dataframe_dictionary = {}
        self.calibrated_files = pd.DataFrame()
        self.peptides_dictionary = {}
        self.linear_regression_model = LinearRegression()

    def run(self) -> None:
        for i, file in tqdm(self.follower_files, desc="Calibrating files"):
            self.__calibrate(file)
    
    def __calibrate(self, follower_file: pd.DataFrame) -> None:
        # set the master file and the follower files
        self.__set_master_follower()
        # get the median for repeated retention times for both the master and follower file
        self.master_file = self.__calculate_median_of_repeated_retention_times(self.master_file)
        follower_file = self.__calculate_median_of_repeated_retention_times(follower_file)
        # filter the files and get anchors
        anchors = self.get_anchors__inner_join(self.master_file, follower_file)
        # train model
        self.__fit_model(anchors['Scan Retention Time_x'].to_numpy().reshape(-1, 1), anchors['Scan Retention Time_y'].to_numpy().reshape(-1, 1))
        # Merge both files and get the transformed retention times for the follower file
        transformed_dataframe = self.__tranform_retention_times(follower_file)
        # Update the dataframe dictionary with the transformed retention times
        self.dataframe_dictionary[follower_file['File Name'].unique()[0]] = follower_file
    
    def __tranform_retention_times(self, retention_times: pd.DataFrame) -> pd.DataFrame:
        retention_times = self.__get_anchors_outer_join(self.master_file, retention_times)
        #make the Transformed Retention Time column
        retention_times['Transformed Retention Time'] = np.nan

        for i in tqdm(range(len(retention_times))):
            if retention_times.loc[i, ['Scan Retention Time_y']].isnull().any() == False and retention_times.loc[i, ['Scan Retention Time_x']].isna().any() == False:
                X = retention_times.loc[i, ['Scan Retention Time_y']].to_numpy().reshape(-1, 1)
                y = self.linear_regression_model.predict(X)
                retention_times.loc[i, ['Transformed Retention Time']] = y.reshape(-1).astype(float)
            
            elif retention_times.loc[i, ['Scan Retention Time_y']].isnull().any() == False and retention_times.loc[i, ['Scan Retention Time_x']].isna().any() == True:
                X = retention_times.loc[i, ['Scan Retention Time_y']].to_numpy().reshape(-1, 1)
                y = self.linear_regression_model.predict(X)
                retention_times.loc[i, ['Transformed Retention Time']] = y.reshape(-1).astype(float)
            
            else:
                retention_times.loc[i, ['Transformed Retention Time']] = retention_times.loc[i, ["Scan Retention Time_x"]].to_numpy().reshape(-1).astype(float)
        
        return retention_times

    def get_calibration_data(self, df: pd.DataFrame) -> pd.DataFrame:
        calibration_data = pd.DataFrame()
        calibration_data['Master'] = df['Scan Retention Time_x']
        calibration_data['Follower'] = df['Scan Retention Time_y']
        return calibration_data
    
    #TODO: Implement this method for the masking of pd.Nan values
    def __replace_nan_values(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def __update_peptides_dictionary(self, transformed_data: pd.DataFrame) -> None:
        full_sequences = dict.fromkeys(transformed_data['Full Sequence'].unique(), [])
        
        for i in tqdm(full_sequences, desc="Updating peptides dictionary"):
            # get the rows that have the full sequence
            rows = transformed_data[transformed_data['Full Sequence'] == i]
            # get the transformed retention time
            transformed = rows['Transformed Retention Time'].values
            # get the retention time x
            x = rows['Scan Retention Time_x'].values
            # get the retention time y
            y = rows['Scan Retention Time_y'].values
            #if there is a nan value in the x or y, replace it with 0
            if np.isnan(x).any() == True:
                x = 0
            if np.isnan(y).any() == True:
                y = 0
            # add the values to the dictionary
            full_sequences[i] = [x, y, transformed]
        
        self.peptides_dictionary = full_sequences

    def __calculate_median_of_repeated_retention_times(self, df: pd.DataFrame) -> pd.DataFrame:
        # get median of repeated retention times
        return df.groupby(['File Name', 'Full Sequence']).agg({'Scan Retention Time': 'median'}).reset_index()

    def __get_anchors_outer_join(self, lead_df: pd.DataFrame, follower_df: pd.DataFrame) -> pd.DataFrame:
        anchors = pd.merge(lead_df, follower_df, on='Full Sequence', how='outer')
        return anchors

    def get_anchors__inner_join(self, master_df: pd.DataFrame, follower_df: pd.DataFrame) -> pd.DataFrame:
        # filter data
        filtered_master_df = self.__filter_file(master_df)
        filtered_follower_df = self.__filter_file(follower_df)
        # get anchors
        anchors = pd.merge(filtered_master_df, filtered_follower_df, on='Full Sequence', how='inner')
        return anchors
    
    def __set_master_follower(self) -> None:
        self.master_file = self.df_list[0]
        self.follower_files = self.df_list[1:]

    def __set_leader_followers(self) -> None:
        self.df_list = [pd.read_csv(path) for path in self.path_list]

    def __get_raw_file_names(self) -> None:
        for i, df in enumerate(self.df_list):
            self.raw_file_names[i] = df['File Name'].unique()

    def __make_dataframes(self) -> None:
        for i, df in enumerate(self.df_list):
            for raw_file in self.raw_file_names[i]:
                self.dataframe_dictionary[raw_file] = df[df['File Name'] == raw_file]

    def __filter_file(self, df: pd.DataFrame) -> pd.DataFrame:
        # fiter by q_value
        df = df[df['q_value'] < 0.01]
        # filter by PEP
        df = df[df['PEP'] < 0.5]
        # filter by ambiguity
        df = df[df['Ambiguity'] == "1"]
        #filter by "Decoy/Contaminant/Target"
        df = df[df['Decoy/Contaminant/Target'] == PsmType.TARGET.value]
        return df
    
    def __fit_model(self, X: np.ndarray, y: np.ndarray) -> None:
        self.linear_regression_model.fit(X, y)

    #TODO: Check how to generalize the sorting of the dictionary
    def show_plot(self, df: pd.DataFrame) -> None:
        plt.clf()

        # sort full peptide discitonary by the transformed retention time
        sorted_full_sequences = {k: v for k, v in sorted(self.peptides_dictionary.items(), key=lambda item: item[1][2])}

        # make a list of all the first values in each key in the dictionary
        x = [v[0] for k, v in sorted_full_sequences.items()]
        # make a list of all the second values in each key in the dictionary
        y = [v[1] for k, v in sorted_full_sequences.items()]
        # make a list of all the third values in each key in the dictionary
        transformed = [v[2] for k, v in sorted_full_sequences.items()]

        # plot the values
        plt.errorbar(range(len(x)), x, linestyle="", c='brown', yerr=0.1, label = "File 1")
        plt.errorbar(range(len(y)), y, linestyle="", c='blue', yerr=0.1, label = "File 2")
        plt.scatter(range(len(transformed)), transformed, s = 1, c='k', label = "Transformed")
        plt.xlabel("Peptide Index")
        plt.ylabel("Retention Time")
        plt.legend()
        #increase plot size
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.show()
        
class PsmType(Enum):
    TARGET = "T"
    DECOY = "D"
    CONTAMINANT = "C"
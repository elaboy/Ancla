import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from tqdm import tqdm
from enum import Enum
from sklearn.base import clone #lets me clone the model parameters, good for tuning parameters and data preprocessing trials
from logger import PeptideLogger

class Calibrator():
    def __init__(self, path_list: list) -> None:
        self.path_list = path_list
        self.master_file = None
        self.follower_files = []
        self.raw_file_names = {}
        self.dataframe_dictionary = {}
        self.calibrated_files = pd.DataFrame()
        self.peptide_loggers = {}
        self.linear_regression_model = LinearRegression()

        self.__prepare_data()

    def __prepare_data(self) -> None:
        # get the raw file names
        for i, df in enumerate(self.path_list):
            loaded_dataframe = pd.read_csv(df, sep='\t')
            # get a list of the file names
            raw_file_name = loaded_dataframe['File Name'].unique()
            for file in raw_file_name:
                #get a dataframe where file name is same as file 
                self.raw_file_names[file] = loaded_dataframe[loaded_dataframe['File Name'] == file]
        
        # get the master file which will be a random dataframe from the raw file names dictionary, master will be removed from the dictionary after being set
        self.master_file = self.raw_file_names.popitem()[1]
        # set the follower files
        self.follower_files = list(self.raw_file_names.values())

        #filter all the dataframes
        # filter the master file         
        # fiter by q_value
        self.master_file = self.master_file[self.master_file['QValue'] < 0.01] 
        self.master_file = self.master_file[self.master_file['PEP'] < 0.5] 
        self.master_file = self.master_file[self.master_file['Ambiguity Level'] == "1"]
        self.master_file = self.master_file[self.master_file['Decoy/Contaminant/Target'] == PsmType.TARGET.value]
        self.master_file = self.master_file.groupby(['File Name', 'Full Sequence']).agg({'Scan Retention Time': 'median'}).reset_index()

        # filter the follower files
        for i in range(len(self.follower_files)):
            self.follower_files[i] = self.follower_files[i][self.follower_files[i]['QValue'] < 0.01]
            self.follower_files[i] = self.follower_files[i][self.follower_files[i]['PEP'] < 0.5] 

            self.follower_files[i] = self.follower_files[i][self.follower_files[i]['Ambiguity Level'] == "1"] 
            self.follower_files[i] = self.follower_files[i][self.follower_files[i]['Decoy/Contaminant/Target'] == PsmType.TARGET.value]
            self.follower_files[i] = self.follower_files[i].groupby(['File Name', 'Full Sequence']).agg({'Scan Retention Time': 'median'}).reset_index()

        for i in range(len(self.follower_files)):
            # get anchors and sort them by the master retention time
            anchors = pd.merge(self.master_file, self.follower_files[i], on='Full Sequence', how='inner').sort_values(by='Scan Retention Time_x')

            # Fit the model
            X = anchors['Scan Retention Time_y'].to_numpy().reshape(-1, 1)
            y = anchors['Scan Retention Time_x'].to_numpy().reshape(-1, 1)

            self.linear_regression_model.fit(X, y)

            # Merge both using the outer join
            transformed_dataframe = pd.merge(self.master_file, self.follower_files[i], on='Full Sequence', how='outer')

            #make the Transformed Retention Time column
            transformed_dataframe['Transformed Retention Time'] = np.nan

            for index in tqdm(range(len(transformed_dataframe)), desc="Transforming retention times"):
                row = transformed_dataframe.iloc[index].to_frame().T

                if row['Scan Retention Time_y'].isnull().values.any() == False and row['Scan Retention Time_x'].isnull().values.any() == False:
                    X = row['Scan Retention Time_y'].to_numpy().reshape(-1, 1)
                    y = self.linear_regression_model.predict(X)
                    transformed_dataframe.loc[index, ['Transformed Retention Time']] = y.reshape(-1).astype(float)
                
                elif row['Scan Retention Time_y'].isnull().values.any() == False and row['Scan Retention Time_x'].isnull().values.any() == True:
                    X = row['Scan Retention Time_y'].to_numpy().reshape(-1, 1)
                    y = self.linear_regression_model.predict(X)
                    transformed_dataframe.loc[index, ['Transformed Retention Time']] = y.reshape(-1).astype(float)
                
                else:
                    transformed_dataframe.loc[index, ['Transformed Retention Time']] = transformed_dataframe.loc[index, ["Scan Retention Time_x"]].to_numpy().reshape(-1).item()
            
            # get all the unique full sequences with all three retention times
            full_sequences = dict.fromkeys(transformed_dataframe['Full Sequence'].unique(), [])

            for full_seq in tqdm(full_sequences, desc="Updating peptides dictionary"):
                #get the follower file name
                file_name = str(self.follower_files[i]['File Name'][0])

                # get the rows that have the full sequence
                rows = transformed_dataframe[transformed_dataframe['Full Sequence'] == full_seq]
                # get the transformed retention time
                transformed = rows['Transformed Retention Time'].to_numpy().item()
                # get the retention time x
                x = rows['Scan Retention Time_x'].to_numpy().item()
                # get the retention time y
                y = rows['Scan Retention Time_y'].to_numpy().item()

                #make the logger object and add it to the dictionary
                if full_seq not in self.peptide_loggers:
                    self.peptide_loggers[full_seq] = PeptideLogger(full_seq)
                    self.peptide_loggers[full_seq].update_file_name_retention_time(file_name, y)
                    self.peptide_loggers[full_seq].update_transformed_retention_times(file_name, transformed)
                else:
                    self.peptide_loggers[full_seq].update_file_name_retention_time(file_name, y)
                    self.peptide_loggers[full_seq].update_transformed_retention_times(file_name, transformed)
            break

    def show_calibration_plot(self) -> None:
        plt.clf()

        retention_times_dataframe = pd.DataFrame(columns=["Full Sequence", "Follower", "Transformed"])

        for k, v in tqdm(self.peptide_loggers.items(), desc="Updating dataframe for plot"):
            retention_times_dataframe.loc[-1] = [k, v.get_retention_times(), v.get_transformed_retention_times()]
            retention_times_dataframe.index = retention_times_dataframe.index + 1
            retention_times_dataframe = retention_times_dataframe.sort_index()

        #sort the dataframe by the transformed retention time
        retention_times_dataframe = retention_times_dataframe.sort_values(by='Transformed').reset_index()

        print(retention_times_dataframe.head())

        #plot the values
        plt.errorbar(range(len(retention_times_dataframe['Follower'])), retention_times_dataframe['Follower'], yerr = 0.8, linestyile = "", c = 'red', label = "Follower")
        plt.scatter(range(len(retention_times_dataframe['Transformed'])), retention_times_dataframe['Transformed'], s = 0.8, c='blue', label = "Transformed")
        plt.xlabel("Peptide Index")
        plt.ylabel("Retention Time")
        plt.legend()
        #increase plot size
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        # show in the plot how many peptides are in the plot for scatter plot
        plt.text(0, 200, f"Follower Peptides: {len(retention_times_dataframe.where(retention_times_dataframe['Follower'].notnull()))}", fontsize=12)
        plt.text(0, 220, f"Transformed Peptides: {len(retention_times_dataframe.where(retention_times_dataframe['Transformed'].notnull()))}", fontsize=12)
        plt.show()
        # # # make a list of all the first values in each key in the dictionary
        # # x = np.array([v.get_retention_times() for k, v in self.peptide_loggers.items()], ndmin=1).reshape(-1)
        # # make a list of all the second values in each key in the dictionary
        # y = np.array([v.get_retention_times() for k, v in self.peptide_loggers.items()], ndmin=1).reshape(-1)
        # # make a list of all the third values in each key in the dictionary
        # transformed = np.array([v.get_transformed_retention_times() for k, v in self.peptide_loggers.items()], ndmin=1).reshape(-1)

        # print(x.shape, y.shape, transformed.shape)

        # #sort the values by the transformed retention time, moving the x and y values with it 
        # # x = [x for _, x in sorted(zip(transformed, x))]
        # y = [y for _, y in sorted(zip(transformed, y))]
        # transformed = sorted(transformed)
        
        # # print(x.shape, y.shape, transformed.shape)

        # # plot the values
        # # plt.errorbar(range(len(x)), x, linestyle="", c='brown', yerr=0.1, label = "File 1")
        # plt.errorbar(range(len(y)), y, linestyle="", c='blue', yerr=0.1, label = "File 2")
        # plt.scatter(range(len(transformed)), transformed, s = 1, c='k', label = "Transformed")
        # plt.xlabel("Peptide Index")
        # plt.ylabel("Retention Time")
        # plt.legend()
        # #increase plot size
        # fig = plt.gcf()
        # fig.set_size_inches(18.5, 10.5)
        # plt.show()

    #TODO: Check how to generalize the sorting of the dictionary
    def show_plot(self) -> None:
        plt.clf()

        # sort full peptide discitonary by the transformed retention time
        sorted_full_sequences = {k: v for k, v in sorted(self.peptides_dictionary.items(), key=lambda item: item[1][2])}

        # make a list of all the first values in each key in the dictionary
        x = np.array([v.get_retention_times() for k, v in sorted_full_sequences.items()], ndmin=1).reshape(-1)
        # make a list of all the second values in each key in the dictionary
        y = np.array([v.get_retention_times() for k, v in sorted_full_sequences.items()], ndmin=1)
        # make a list of all the third values in each key in the dictionary
        transformed = np.array([v.get_transformed_retention_times() for k, v in sorted_full_sequences.items()])

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
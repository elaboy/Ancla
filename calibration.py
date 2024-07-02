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

        for i in tqdm(range(len(self.follower_files)), desc="Calibrating file"):
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
                    self.peptide_loggers[full_seq].update_master_file_name_retention_time(file_name, x)
                    self.peptide_loggers[full_seq].update_file_name_retention_time(file_name, y)
                    self.peptide_loggers[full_seq].update_transformed_retention_times(file_name, transformed)
                else:
                    self.peptide_loggers[full_seq].update_master_file_name_retention_time(file_name, x)
                    self.peptide_loggers[full_seq].update_file_name_retention_time(file_name, y)
                    self.peptide_loggers[full_seq].update_transformed_retention_times(file_name, transformed)

    def show_calibration_plot(self) -> None:
        plt.clf()

        # retention_times_dataframe = pd.DataFrame(columns=["Full Sequence", "Master", "Follower", "Transformed"])
        rows_to_add = []
        for k, v in tqdm(self.peptide_loggers.items(), desc="Updating dataframe for plot"):
            # for each peptide logger, get the file name and the retention times
            for file_name in v.get_file_names_from_retention_time():
                rows_to_add.append([file_name, k, v.get_file_name_retention_time()[file_name],
                                     v.get_file_name_retention_time()[file_name], v.get_transformed_retention_time()[file_name]])
            
            for transformed_file_name in v.get_transformed_file_names_from_retention_time():
                rows_to_add.append([transformed_file_name, k, v.get_file_name_retention_time()[transformed_file_name],
                                     v.get_file_name_retention_time()[transformed_file_name], v.get_transformed_retention_time()[transformed_file_name]])

            # retention_times_dataframe.loc[-1] = [k, v.get_master_retention_times(), v.get_retention_times(), v.get_transformed_retention_times()]
            # retention_times_dataframe.index = retention_times_dataframe.index + 1
            # retention_times_dataframe = retention_times_dataframe.sort_index()

        retention_times_dataframe = pd.DataFrame(rows_to_add, columns=["File Name", "Full Sequence", "Master", "Follower", "Transformed"])

        # separate the dataframes by the file name
        dfs = [retention_times_dataframe[retention_times_dataframe['File Name'] == file_name] for file_name in retention_times_dataframe['File Name'].unique()]
        
        peptides_present = retention_times_dataframe['Full Sequence'].unique()
        
        # sort the dataframes by the transformed retention time
        for i in range(len(dfs)):
            dfs[i] = dfs[i].sort_values(by='Transformed').reset_index()

        from logger import File
        files = []
        for df in dfs:
            file = File()
            file.make_keys(peptides_present)
            #drop full sequences that are identical
            df = df.drop_duplicates(subset='Full Sequence')
            for index, row in tqdm(df.iterrows()):
                # follower = row['Follower'] if row['Follower'] != None else np.nan
                transformed = row['Transformed'] if row['Transformed'] != None else np.nan
            
                file.update_full_sequence_times(row['Full Sequence'], transformed)

            files.append(file)

        # plot file one dictionary
        dictionary = files[0].full_sequence_times

        import collections

        # sort the doctionary by the transformed retention time
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[1])
        dictionary = collections.OrderedDict(sorted_dict)        
        values = list(dictionary.values())
        plt.scatter(range(len(dictionary.keys())), dictionary.items(), linestyle="-", c='brown', label = "File 1")
        plt.xlabel("Peptide Index")
        plt.ylabel("Retention Time")
        plt.show()

        # plt.show()
        # ranked_full_sequences = [df['Full Sequence'].unique() for df in dfs]

        # # plot the values
        # for index, full_sequence in enumerate(ranked_full_sequences):
        #     for df_index, df in enumerate(dfs):
        #         match = df[df['Full Sequence'] == full_sequence[df_index]]
        #         master = match['Master'].to_numpy()
        #         follower = match['Follower'].to_numpy()
        #         transformed = match['Transformed'].to_numpy()
                
        #         # get one value for each full sequence
        #         master = np.median(master).item()
        #         follower = np.median(follower).item()
        #         transformed = np.median(transformed).item()

        #         plt.scatter(index, master, c='brown', label = "Master")
        #         plt.scatter(index, follower, c='blue', label = "Follower")
        #         plt.scatter(index, transformed, c='k', label = "Transformed")
        
        # plt.show()
                

        # # merge dataframes into one where they are joined by file name and the full sequence
        # plt.xlabel("Peptide Index")
        # plt.ylabel("Retention Time")
        # plt.legend()

        # # legends outside the plot
        # # axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # # axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # plt.viridis()
        # plt.xlabel("Peptide Index")
        # plt.ylabel("Retention Time")
        # plt.legend()
        # #increase plot size
        # plt.show()
        # # change nan values to 0

        # print(retention_times_dataframe.head())

        # split Master, Follower and Transformed into 18 columns each, the cells have list of vlaues that should be unpacked
        # retention_times_dataframe = pd.concat([retention_times_dataframe['Full Sequence'], retention_times_dataframe['Master'].apply(pd.Series),
        #                                         retention_times_dataframe['Follower'].apply(pd.Series), retention_times_dataframe['Transformed'].apply(pd.Series)], axis=1)
        # # rename the columns
        # retention_times_dataframe.columns = ['Full Sequence', 'Master_1', 'Master_2', 'Master_3', 'Master_4', 'Master_5', 'Master_6', 'Master_7', 'Master_8', 'Master_9', 'Master_10',
        #                                       'Master_11', 'Master_12', 'Master_13', 'Master_14', 'Master_15', 'Master_16', 'Master_17', 'Follower_1', 'Follower_2', 'Follower_3', 'Follower_4',
        #                                         'Follower_5', 'Follower_6', 'Follower_7', 'Follower_8', 'Follower_9', 'Follower_10', 'Follower_11', 'Follower_12', 'Follower_13', 'Follower_14',
        #                                           'Follower_15', 'Follower_16', 'Follower_18', 'Transformed_1', 'Transformed_2', 'Transformed_3', 'Transformed_4', 'Transformed_5', 'Transformed_6',
        #                                             'Transformed_7', 'Transformed_8', 'Transformed_9', 'Transformed_10', 'Transformed_11', 'Transformed_12', 'Transformed_13', 'Transformed_14',
        #                                               'Transformed_15', 'Transformed_16', 'Transformed_17']

        # merge master, follower and transformed columns where their values will be the median of the values in the columns
        # retention_times_dataframe['Master'] = retention_times_dataframe[['Master_1', 'Master_2', 'Master_3', 'Master_4', 'Master_5', 'Master_6', 'Master_7', 'Master_8', 'Master_9', 'Master_10', 'Master_11', 'Master_12', 'Master_13', 'Master_14', 'Master_15', 'Master_16', 'Master_17']].median(axis=1)
        # retention_times_dataframe['Follower'] = retention_times_dataframe[['Follower_1', 'Follower_2', 'Follower_3', 'Follower_4', 'Follower_5', 'Follower_6', 'Follower_7', 'Follower_8', 'Follower_9', 'Follower_10', 'Follower_11', 'Follower_12', 'Follower_13', 'Follower_14', 'Follower_15', 'Follower_16', 'Follower_18']].median(axis=1)
        # retention_times_dataframe['Transformed'] = retention_times_dataframe[['Transformed_1', 'Transformed_2', 'Transformed_3', 'Transformed_4', 'Transformed_5', 'Transformed_6', 'Transformed_7', 'Transformed_8', 'Transformed_9', 'Transformed_10', 'Transformed_11', 'Transformed_12', 'Transformed_13', 'Transformed_14', 'Transformed_15', 'Transformed_16', 'Transformed_17']].median(axis=1)

        # sort by Transformed_1
        # retention_times_dataframe = retention_times_dataframe.sort_values(by='Transformed_17').reset_index()

        # # plot the values
        # plt.plot(range(len(retention_times_dataframe['Master'])), retention_times_dataframe['Master'], linestyle="-", c='brown', label = "Master")
        # plt.errorbar(range(len(retention_times_dataframe['Master_1'])), retention_times_dataframe['Master_1'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5, label = "Master")
        # plt.errorbar(range(len(retention_times_dataframe['Follower_1'])), retention_times_dataframe['Follower_1'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5, label = "Follower")
        # plt.scatter(range(len(retention_times_dataframe['Transformed_1'])), retention_times_dataframe['Transformed_1'], s = 0.3, linestyle = "", c='navy', label = "Transformed")

        # plt.errorbar(range(len(retention_times_dataframe['Master_2'])), retention_times_dataframe['Master_2'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_2'])), retention_times_dataframe['Follower_2'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_2'])), retention_times_dataframe['Transformed_2'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_3'])), retention_times_dataframe['Master_3'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_3'])), retention_times_dataframe['Follower_3'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_3'])), retention_times_dataframe['Transformed_3'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_4'])), retention_times_dataframe['Master_4'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_4'])), retention_times_dataframe['Follower_4'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_4'])), retention_times_dataframe['Transformed_4'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_5'])), retention_times_dataframe['Master_5'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_5'])), retention_times_dataframe['Follower_5'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_5'])), retention_times_dataframe['Transformed_5'], s = 0.3, linestyle = "", c='navy')
        
        # plt.errorbar(range(len(retention_times_dataframe['Master_6'])), retention_times_dataframe['Master_6'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_6'])), retention_times_dataframe['Follower_6'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_6'])), retention_times_dataframe['Transformed_6'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_7'])), retention_times_dataframe['Master_7'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_7'])), retention_times_dataframe['Follower_7'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_7'])), retention_times_dataframe['Transformed_7'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_8'])), retention_times_dataframe['Master_8'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_8'])), retention_times_dataframe['Follower_8'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_8'])), retention_times_dataframe['Transformed_8'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_9'])), retention_times_dataframe['Master_9'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_9'])), retention_times_dataframe['Follower_9'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_9'])), retention_times_dataframe['Transformed_9'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_10'])), retention_times_dataframe['Master_10'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_10'])), retention_times_dataframe['Follower_10'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_10'])), retention_times_dataframe['Transformed_10'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_11'])), retention_times_dataframe['Master_11'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_11'])), retention_times_dataframe['Follower_11'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_11'])), retention_times_dataframe['Transformed_11'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_12'])), retention_times_dataframe['Master_12'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_12'])), retention_times_dataframe['Follower_12'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_12'])), retention_times_dataframe['Transformed_12'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_13'])), retention_times_dataframe['Master_13'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_13'])), retention_times_dataframe['Follower_13'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_13'])), retention_times_dataframe['Transformed_13'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_14'])), retention_times_dataframe['Master_14'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_14'])), retention_times_dataframe['Follower_14'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_14'])), retention_times_dataframe['Transformed_14'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_15'])), retention_times_dataframe['Master_15'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_15'])), retention_times_dataframe['Follower_15'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_15'])), retention_times_dataframe['Transformed_15'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_16'])), retention_times_dataframe['Master_16'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_16'])), retention_times_dataframe['Follower_16'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_16'])), retention_times_dataframe['Transformed_16'], s = 0.3, linestyle = "", c='navy')

        # plt.errorbar(range(len(retention_times_dataframe['Master_17'])), retention_times_dataframe['Master_17'], yerr = 0.2, linestyle = "", c = 'lightcoral', alpha=0.5)
        # plt.errorbar(range(len(retention_times_dataframe['Follower_18'])), retention_times_dataframe['Follower_18'], yerr = 0.2, linestyle = "", c = 'wheat', alpha=0.5)
        # plt.scatter(range(len(retention_times_dataframe['Transformed_17'])), retention_times_dataframe['Transformed_17'], s = 0.3, linestyle = "", c='navy')
        # #save the dataframe
        # retention_times_dataframe.to_csv("calibrated_data.csv", index=False)
        # # melt the dataframe
        # retention_times_dataframe = pd.melt(retention_times_dataframe, id_vars=['Full Sequence'], var_name='Type', value_name='Retention Time')
        # # sort the dataframe by the transformed retention time
        # retention_times_dataframe = retention_times_dataframe.sort_values(by='Retention Time').reset_index()


        # retention_times_dataframe = retention_times_dataframe.fillna(0)
        # # if any string values are present, change them to 0
        # retention_times_dataframe = retention_times_dataframe.apply(pd.to_numeric, errors='coerce').fillna(0)
        # retention_times_dataframe = retention_times_dataframe.sort_values(by='Transformed').reset_index()


        # #plot the values
        # plt.errorbar(range(len(retention_times_dataframe['Master'])), retention_times_dataframe['Master'], yerr = 0.2, linestyle = "", c = 'gray', alpha=0.5, label = "Master")
        # plt.errorbar(range(len(retention_times_dataframe['Follower'])), retention_times_dataframe['Follower'], yerr = 0.2, linestyle = "", c = 'gray', alpha=0.5, label = "Follower")
        # plt.scatter(range(len(retention_times_dataframe['Transformed'])), retention_times_dataframe['Transformed'], s = 0.3, linestyle = "", c='red', label = "Transformed")
        # plt.xlabel("Peptide Index")
        # plt.ylabel("Retention Time")
        # # #increase plot size
        # # fig = plt.gcf()
        # # fig.set_size_inches(18.5, 10.5)
        # # #number of peptides
        # # plt.text(5000, 0, f"Number of Peptides: {len(retention_times_dataframe)}", fontsize=12)
        # plt.show()
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
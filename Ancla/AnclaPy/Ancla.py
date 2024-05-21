import stat
from statistics import LinearRegression
from numpy._typing import NDArray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peptides
import torch

aa_dict = {
    "PAD" : 0,
    "A" : 1,
    "R" : 2,
    "N" : 3,
    "D" : 4,
    "C" : 5,
    "E" : 6,
    "Q" : 7,
    "G" : 8,
    "H" : 9,
    "I" : 10,
    "L" : 11,
    "K" : 12,
    "M" : 13,
    "F" : 14,
    "P" : 15,
    "S" : 16,
    "T" : 17,
    "W" : 18,
    "Y" : 19,
    "V" : 20,
    "U" : 21,
}

class Featurizer(object):
    @staticmethod
    def featurize(data: str) -> NDArray:
        # the array will be of shape (peptide, residues, features)

        # split the string into a list of characters
        peptides_as_char_list = [list(data)]

        # add PAD until the length is 100
        padded_peptides_char_list = [peptide + ["PAD"] * (100 - len(peptide)) for peptide in peptides_as_char_list]

        # conver data into a list of peptides where each character is replaces by its corresponding value in the aa_dict
        peptides_as_dict_index = [[aa_dict[aa] for aa in peptide] for peptide in padded_peptides_char_list]
        
        # compute data features per residue 
        hydrophobicity = [peptides.Peptide(residue).hydrophobic_moment() for residue in data]
        
        #pad the hydrophobicity array with zeros until the length is 100
        hydrophobicity = [hydrophobicity + [0] * (100 - len(hydrophobicity))]
        
        # compute the z-scales
        z_scales = [peptides.Peptide(residue).z_scales() for residue in data]
        
        #pad the z-scales array with zeros until the length is 100
        z_scales = [z_scales + [0] * (100 - len(z_scales))]

        z1_scale, z2_scale, z3_scale, z4_scale, z5_scale = [], [], [], [], []

        for z_scale in z_scales:
            for i, z in enumerate(z_scale):
                if z != 0:
                    z1_scale.append(z.z1)
                    z2_scale.append(z.z2)
                    z3_scale.append(z.z3)
                    z4_scale.append(z.z4)
                    z5_scale.append(z.z5)
                else:
                    z1_scale.append(0)
                    z2_scale.append(0)
                    z3_scale.append(0)
                    z4_scale.append(0)
                    z5_scale.append(0)

        # compute the charge
        charge = [peptides.Peptide(residue).charge() for residue in data]

        #pad the charge array with zeros until the length is 100
        charge = [charge + [0] * (100 - len(charge))]

        # add the features into the final array, vstacked
        final_array = np.vstack(
            [np.array(peptides_as_dict_index), 
             np.array(hydrophobicity),
             np.array(z1_scale),
             np.array(z2_scale),
             np.array(z3_scale),
             np.array(z4_scale),
             np.array(z5_scale), 
             np.array(charge)]
            )

        return final_array
    
    @staticmethod
    def featurize_all(data: list) -> NDArray:
        # parallelize the featurization of the data
        from joblib import Parallel, delayed

        # featurize the data
        featurized_data = Parallel(n_jobs=-1)(delayed(Featurizer.featurize)(data) for data in data)

        return np.stack(featurized_data)

    @staticmethod
    def min_max_scaler(data: NDArray) -> NDArray:
        # Assuming 'data' is your 3D array
        mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        std_dev = np.std(data, axis=(0, 1, 2), keepdims=True)

        # Avoid division by zero
        std_dev = np.where(std_dev == 0, 1e-9, std_dev)

        # Apply feature-wise normalization
        normalized_data = (data - mean) / std_dev

        return normalized_data
    
    @staticmethod
    def featurize_all_normalized(data: list) -> NDArray:
        return Featurizer.min_max_scaler(Featurizer.featurize_all(data))

class HelperFunctions(object):
    
    #read data from csv. Headers are FullSequence, Database, Experimental, PostTransformation
    @staticmethod
    def read_results(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        return data
    
    @staticmethod
    def save_results(data: pd.DataFrame, path: str) -> None:
        data.to_csv(path, index=False)

    @staticmethod
    def standardScaler(data: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import StandardScaler
        
        #create the scaler
        scaler = StandardScaler()
        
        #fit the scaler and transform the columns Database, Experimental and PostTransformation
        data[['Database', 'Experimental', 'PostTransformation']] = scaler.fit_transform(data[['Database', 'Experimental', 'PostTransformation']])

    @staticmethod
    def reject_outliers(data: pd.DataFrame) -> pd.DataFrame:
        from scipy.stats import zscore
        
        # reject outliers from the Database and Experimental columns
        data['Database_z'] = zscore(data['Database'])
        data['Experimental_z'] = zscore(data['Experimental'])

        #reject the outliers
        data = data[(np.abs(data['Database_z']) < 3) & (np.abs(data['Experimental_z']) < 3)]

        #drop the z-score columns
        data = data.drop(columns=['Database_z', 'Experimental_z'])

        return data
        
class Visualize(object):
    
    @staticmethod
    def scatter_plots(data: pd.DataFrame) -> None:
        """ Return None, only shows the plots.
        
        Three scatter plots: Database vs Database, Experimental vs Database, PostTransformation vs Database
        """
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
        from sklearn.metrics import r2_score

        #first plot
        axs[0].scatter(data['Database'], data['Database'], 
                       color='red', label='Database vs Database', s=0.3)
        axs[0].scatter(data['Experimental'], data['Database'], 
                       color='blue', label='Experimental vs Database', s=0.3)
        axs[0].set_xlabel('Experimental Peptide RT')
        axs[0].set_ylabel('Database Peptide RT')
        axs[0].legend()
        
        #second plot
        axs[1].scatter(data['Database'], data['Database'],
                       color='red', label='Database vs Database', s=0.3)
        axs[1].scatter(data['Experimental'], data['Database'],
                       color='blue', label='Experimental vs Database', s=0.3)
        axs[1].scatter(data['PostTransformation'], data['Database'],
                       color='green', label='PostTransformation vs Database', s=0.3)
        axs[1].set_xlabel('PostTransformation')
        axs[1].set_ylabel('Database')
        axs[1].legend()
    
        #dd R2 score to the plot
        r2_exp = r2_score(data['Database'], data['Experimental'])
        axs[0].text(0.5, 0.9, f'R2: {r2_exp:.2f}', transform=axs[0].transAxes)

        r2 = r2_score(data['Database'], data['PostTransformation'])
        axs[1].text(0.5, 0.9, f'R2: {r2:.2f}', transform=axs[1].transAxes)
        
    @staticmethod
    def z_scores_hist(data: pd.DataFrame) -> None:
        from scipy.stats import zscore
        
        #compute the z-scores for the Database and Experimental columns, then database and PostTransformation
        data['Database_z'] = zscore(data['Database'])
        data['Experimental_z'] = zscore(data['Experimental'])
        data['PostTransformation_z'] = zscore(data['PostTransformation'])
        
        #show the z-scores before and after the transformation. I want to see if the distribution is the same
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        axs[0].hist(data['Database_z'], bins=100, color='red', alpha=0.5, label='Database')
        axs[0].hist(data['Experimental_z'], bins=100, color='blue', alpha=0.5, label='Experimental')
        axs[0].set_title('Before transformation')
        axs[0].set_xlabel('Z-score')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        
        axs[1].hist(data['Database_z'], bins=100, color='red', alpha=0.5, label='Database')
        axs[1].hist(data['PostTransformation_z'], bins=100, color='green', alpha=0.5, label='PostTransformation')
        axs[1].set_title('After transformation')
        axs[1].set_xlabel('Z-score')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        
        #add trendline to the histograms
        from scipy.stats import norm
        from scipy import stats
       
        #fit a normal distribution to the data
        mu, std = norm.fit(data['Database_z'])
        x = np.linspace(-5, 5, 100)
        p = norm.pdf(x, mu, std)
        
        axs[0].plot(x, p, 'k', linewidth=2)
        axs[1].plot(x, p, 'k', linewidth=2)

    @staticmethod
    def residuals_scatter(data: pd.DataFrame) -> None:
        #compute the residuals for the experimental and post transformation columns
        data['Residuals_exp'] = data['Database'] - data['Experimental']
        data["Residuals_trans"] = data['Database'] - data['PostTransformation']
        
        #plot the data and the residuals in two plots in the same grid 
        fig, axs = plt.subplots(2, 2, figsize=(15, 5))
        
        axs[0, 0].scatter(data['Database'], data['Database'],
                          color='red', label='Database vs Database', s=0.3)
        axs[0, 0].scatter(data['Experimental'], data['Database'],
                          color='blue', label='Experimental vs Database', s=0.3)
        axs[0, 0].set_xlabel('Experimental')
        axs[0, 0].set_ylabel('Database')
        axs[0, 0].legend()
        
        axs[0, 1].scatter(range(len(data["Residuals_exp"])), data['Residuals_exp'], 
                          color='blue', label='Residuals Experimental', s=0.3)
        #horizontal black line at 0
        axs[0, 1].axhline(0, color='black', lw=1)
        axs[0, 1].set_xlabel('Residuals Experimental')
        axs[0, 1].set_ylabel('Database')
        axs[0, 1].legend()
        
        axs[1, 0].scatter(data['Database'], data['Database'],
                          color='red', label='Database vs Database', s=0.3)
        axs[1, 0].scatter(data['PostTransformation'], data['Database'],
                          color='green', label='PostTransformation vs Database', s=0.3)
        axs[1, 0].set_xlabel('PostTransformation')
        axs[1, 0].set_ylabel('Database')
        axs[1, 0].legend()
        
        axs[1, 1].scatter(range(len(data["Residuals_trans"])), data["Residuals_trans"],
                          color='green', label='Residuals PostTransformation', s=0.3)
        #horizontal black line at 0
        axs[1, 1].axhline(0, color='black', lw=1)
        axs[1, 1].set_xlabel('Residuals PostTransformation')
        axs[1, 1].set_ylabel('Database')
        axs[1, 1].legend()
        

    @staticmethod
    def qq_plot(data: pd.DataFrame) -> None:
        from scipy.stats import probplot
        
        #compute the residuals for the experimental and post transformation columns
        data['Residuals_exp'] = data['Database'] - data['Experimental']
        data["Residuals_trans"] = data['Database'] - data['PostTransformation']
        
        #plot the qq plot for the residuals
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        probplot(data['Residuals_exp'], plot=axs[0])
        axs[0].set_title('QQ plot for the residuals of the Experimental column')
        
        probplot(data['Residuals_trans'], plot=axs[1])
        axs[1].set_title('QQ plot for the residuals of the PostTransformation column')

    @staticmethod
    def visualize_all(data: pd.DataFrame) -> None:
        Visualize.scatter_plots(data)
        Visualize.z_scores_hist(data)
        Visualize.residuals_scatter(data)
        # Visualize.qq_plot(data)
        plt.show()

    @staticmethod
    def save_visualize_fractions(data: pd.DataFrame, path: str) -> None:
        from sklearn.metrics import r2_score

        Visualize.scatter_plots(data)
        plt.savefig(path + "/_scatter_plots.png")
        plt.clf()
        Visualize.z_scores_hist(data)
        plt.savefig(path + "/_z_scores_hist.png")
        plt.clf()
        Visualize.residuals_scatter(data)
        plt.savefig(path + "/_residuals_scatter.png")
        # Visualize.qq_plot(data).savefig(path + "_qq_plot.png")
        plt.clf()

class RegressiveModels(object):
    @staticmethod
    def fit_linear(data: pd.DataFrame, x: str="Experimental", y: str="Database") -> NDArray:
        from sklearn.linear_model import LinearRegression
    
        model = LinearRegression()
    
        #fit the model
        model.fit(data[x].values.reshape(-1, 1), data[y])
    
        return model

    @staticmethod
    def fit_lowess(data: pd.DataFrame, x: str="Experimental", y: str="Database", 
                  frac: float=1./3) -> NDArray:
    
        from statsmodels.nonparametric.smoothers_lowess import lowess
    
        # order the dataframe by the Database column
        data = data.sort_values(by=[y])

        #take the values of the columns
        database = data[y].to_list()
        experimental = data[x].to_list()
    
        #fit the lowess model
        model = lowess(database, experimental, frac = frac, it = 4)
    
        return model

class TransformationFunctions(object):
    @staticmethod    
    def transform_experimental_lowess(data: pd.DataFrame, lowess_model: NDArray,
                               x: str="Experimental", y: str="Database") -> pd.DataFrame:
        #transform the experimental values
        data['PostTransformation'] = np.interp(data['Experimental'], lowess_model[:, 0], lowess_model[:, 1])
    
        return data
    
    @staticmethod
    def transform_experimental_linear(data: pd.DataFrame, linear_model: LinearRegression,
                               x: str="Experimental", y: str="Database") -> pd.DataFrame:
        #transform the experimental values
        data['PostTransformation'] = linear_model.predict(data[x].values.reshape(-1, 1))
    
        return data
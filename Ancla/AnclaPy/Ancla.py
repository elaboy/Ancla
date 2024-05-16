import stat
from statistics import LinearRegression
from numpy._typing import NDArray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class HelperFunctions(object):
    
    #read data from csv. Headers are FullSequence, Database, Experimental, PostTransformation
    @staticmethod
    def read_results(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        return data
    
    @staticmethod
    def save_results(data: pd.DataFrame, path: str) -> None:
        data.to_csv(path, index=False)

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
        axs[0].set_xlabel('Experimental')
        axs[0].set_ylabel('Database')
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
        Visualize.qq_plot(data)
        plt.show()

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
                  frac: float=0.1) -> NDArray:
    
        from statsmodels.nonparametric.smoothers_lowess import lowess
    
        # order the dataframe by the Database column
        data = data.sort_values(by=[y])

        #take the values of the columns
        database = data['Database'].to_list()
        experimental = data['Experimental'].to_list()
    
        #fit the lowess model
        model = lowess(database, experimental, frac=1./3, it=4)
    
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
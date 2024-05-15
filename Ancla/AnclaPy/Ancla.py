from numpy._typing import NDArray
import pandas as pd
import numpy as np

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
    import matplotlib.pyplot as plt
    
    @staticmethod
    def scatter_plots(data: pd.DataFrame) -> None:
        """ Return None, only shows the plots.
        
        Three scatter plots: Database vs Database, Experimental vs Database, PostTransformation vs Database
        """
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
        from sklearn.metrics import r2_score

        #first plot
        axs[0].scatter(data['Database'], data['Database'], color='red', label='Database vs Database', s=0.3)
        axs[0].scatter(data['Experimental'], data['Database'], color='blue', label='Experimental vs Database', s=0.3)
        axs[0].set_xlabel('Experimental')
        axs[0].set_ylabel('Database')
        axs[0].legend()
        
        #second plot
        axs[1].scatter(data['Database'], data['Database'], color='red', label='Database vs Database', s=0.3)
        axs[1].scatter(data['Experimental'], data['Database'], color='blue', label='Experimental vs Database', s=0.3)
        axs[1].scatter(data['PostTransformation'], data['Database'], color='green', label='PostTransformation vs Database', s=0.3)
        axs[1].set_xlabel('PostTransformation')
        axs[1].set_ylabel('Database')
        axs[1].legend()
    
        #dd R2 score to the plot
        r2_exp = r2_score(data['Database'], data['Experimental'])
        axs[0].text(0.5, 0.9, f'R2: {r2_exp:.2f}', transform=axs[0].transAxes)

        r2 = r2_score(data['Database'], data['PostTransformation'])
        axs[1].text(0.5, 0.9, f'R2: {r2:.2f}', transform=axs[1].transAxes)

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
        model = lowess(database, experimental, frac=1./3)
    
        return model

class TransformationFunctions(object):
    @staticmethod    
    def transform_experimental_lowess(data: pd.DataFrame, lowess_model: NDArray,
                               x: str="Experimental", y: str="Database") -> pd.DataFrame:
        #transform the experimental values
        data['PostTransformation'] = np.interp(data['Experimental'], lowess_model[:, 0], lowess_model[:, 1])
    
        return data

if __name__ == '__main__':
    data = read_results(r'D:\transformedData_RAW.csv')
    model = fit_lowess(data)
    data = transform_experimental(data, model)
    scatter_plots(data)
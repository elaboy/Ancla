import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#read data from csv. Headers are FullSequence, Database, Experimental, PostTransformation
def read_results(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

#Three scatter plots: Database vs Database, Experimental vs Database, PostTransformation vs Database
def scatter_plots(data: pd.DataFrame) -> pd.DataFrame:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.scatterplot(data=data, x='Database', y='Database', ax=axs[0])
    sns.scatterplot(data=data, x='Database', y='Experimental', ax=axs[1])
    sns.scatterplot(data=data, x='Database', y='PostTransformation', ax=axs[2])
    plt.show()
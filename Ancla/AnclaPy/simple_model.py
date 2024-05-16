from webbrowser import get
from numpy.typing import NDArray
import torch 
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#import csv as panda dataframe 
data = pd.read_csv('D:\\PSMs_RAW.csv')

def get_encoded_data(data: pd.DataFrame) -> (NDArray, list, OneHotEncoder):
    #take all the full sequences and retentions times
    X = data['BaseSequence']
    y = data['ScanRetentionTime']

    # aminoacids with a number to represent them, list within list
    aminoacids = [["PAD", 0], ['A', 1], ['R', 2], ['N', 3], ['D', 4],
                  ['C', 5,], ['Q', 6], ['E', 7], ['G', 8], ['H', 9],
                  ['I', 10], ['L', 11],['K', 12],['M', 13],['F', 14],
                  ['P', 15], ['S', 16],['T', 17],['W', 18],['Y', 19],['V', 20],
                  ["U", 21]]


    #convert the sequences to one hot encoding 
    enc = OneHotEncoder(handle_unknown='ignore')
    oneHotEncoderFitted = enc.fit(aminoacids)
    dataset = []
    
    for sequence in X:
        example = []
        
        for residue in sequence:
            #get residue numbeer form aminoacid list
            print(residue)
            residue_num = [x for x in aminoacids if x[0] == residue]
    
            example.append(oneHotEncoderFitted.transform(residue_num).toarray())

        while len(example) < 65:
            example.append(oneHotEncoderFitted.transform([["PAD", 0]]).toarray())

        dataset.append(example)
        
    dataset = np.stack(dataset)
    
    return (dataset, y, oneHotEncoderFitted)

if __name__ == "__main__":
    dataset, retention_times, encoder = get_encoded_data(data)
    print(dataset.shape)
    print(len(retention_times))
    
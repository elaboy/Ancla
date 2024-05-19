from importlib import simple
from token import OP
from webbrowser import get

import statsmodels.regression.mixed_linear_model
from numpy.lib.index_tricks import AxisConcatenator
from numpy.typing import NDArray
import torch 
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

#import csv as panda dataframe 
# data = pd.read_csv(r"\192.168.1.115\nas\MSData\PSMs_RAW.csv")
data = pd.read_csv("/Volumes/NAS/MSData/PSMs_RAW.csv")
aa_dictionary = {
    "PAD": 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4,
    "C": 5, "Q": 6, "E": 7, "G": 8, "H": 9,
    "I": 10, "L": 11, "K": 12, "M": 13, "F": 14,
    "P": 15, "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20,
    "U" : 21}

def get_tokens(data: pd.DataFrame) -> (list, list):
    base_sequences = data['BaseSequence']
    retention_times = data["ScanRetentionTime"]
    
    tokens = []
    
    for sequence in base_sequences:
        sequence_tokens = []
        
        for residue in sequence:
            sequence_tokens.append(aa_dictionary[residue])
        
        while len(sequence_tokens) != 100:
            sequence_tokens.append(0)
        tokens.append(sequence_tokens)

    return (tokens, retention_times)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.cnn = nn.Conv1d(32, 100, 3)
        self.cnn2 = nn.Conv1d(100, 100, 3)
        self.cnn3 = nn.Conv1d(100, 100, 3)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(94, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        
        self.normalize = nn.LayerNorm(32)
        self.dropout = nn.Dropout(0.5)
        
        self.double()
        
    def forward(self, x):
        # x = self.embedding(x)
        # x = self.normalize(x)
        # x = x.permute(1, 0, 2)
        # x, _ = self.attention(x, x, x)
        x = self.cnn(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        # x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x
    
    def train_model(model, train_loader, optimizer, criterion, epochs):
        model.train()
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                # print(x.shape, y.shape)
                optimizer.zero_grad()
                y_pred = model(x.to("cuda" if torch.cuda.is_available() else "cpu"))
                loss = criterion(y_pred, y.to("cuda" if torch.cuda.is_available() else "cpu"))
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch}, Loss: {loss.item()}, Step: {i} of {len(train_loader)}")
                
    def test_model(model, test_loader):
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                y_pred = model(x)
                print(f"Prediction: {y_pred}, Actual: {y}")
    
    def validate_model(model, val_loader, criterion):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for x, y in val_loader:
                y_pred = model(x.to("cuda" if torch.cuda.is_available() else "cpu"))
                loss = criterion(y_pred, y.to("cuda" if torch.cuda.is_available() else "cpu"))
                total_loss += loss.item()
            print(f"Validation Loss: {total_loss}")
            
    def train_validate(model, train_loader, val_loader, optimizer, criterion, epochs):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        for epoch in range(epochs):
            for i, (x, y) in enumerate(train_loader):
                model.train()
                
                optimizer.zero_grad()
                
                y_pred = model(x.to("cuda" if torch.cuda.is_available() else "cpu"))
                loss = criterion(y_pred, y.to("cuda" if torch.cuda.is_available() else "cpu"))
                
                loss.backward()
                optimizer.step()
                
                # print every 100 steps
                if i % 100 == 0:
                    print(f"Epoch: {epoch}, Loss: {loss.item()}, Step: {i} of {len(train_loader)}")
                    
            SimpleModel.validate_model(model, val_loader, criterion)
            scheduler.step()
            

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame):
        self.full_sequence, self.retention_times = get_tokens(data)
    
    def __len__(self):
        return len(self.full_sequence)
    
    def __getitem__(self, idx):
        return torch.tensor(self.full_sequence[idx], dtype=torch.double), torch.tensor(self.retention_times[idx])

if __name__ == "__main__":
    #shuffle data 
    data = data.sample(frac=1).reset_index(drop=True)
    
    dataset = get_tokens(data)
    # manually split the data into train and test, whitout the use of any package
    dataset_size = len(dataset[0])
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    
    #get training datasframe and test dataframe from dataset
    train_df = pd.DataFrame({"BaseSequence": dataset[0][:train_size], "ScanRetentionTime": dataset[1][:train_size]})
    test_df = pd.DataFrame({"BaseSequence": dataset[0][train_size:], "ScanRetentionTime": dataset[1][train_size:]})
    
    train_dataset = train_df.iloc[:, 0], train_df.iloc[:, 1]
    train_dataset = (np.array(train_dataset[0].tolist()), np.array(train_dataset[1].tolist()))
    
    test_dataset = test_df.iloc[:, 0], test_df.iloc[:, 1]
    test_dataset = (np.array(test_dataset[0].tolist()), np.array(test_dataset[1].tolist()))
    
    
    X = train_dataset[0]
    y = train_dataset[1]
    # 
    # #split y into a list of (1,) arrays
    # y = np.array([np.array([i]) for i in y])
    #normalize the data
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X = sc.fit_transform(X)
    # y = sc.fit_transform(y)
    
    #split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # linear regression model from sklearn
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    #predict the test set results
    y_pred = regressor.predict(X_test)
    
    #plot the results
    import matplotlib.pyplot as plt
    plt.scatter(y_pred, y_test, color='red')
    plt.title('Retention Time vs Base Sequence')
    plt.xlabel('Base Sequence')
    plt.ylabel('Retention Time')
    plt.show()


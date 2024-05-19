from importlib import simple
from token import OP
from webbrowser import get
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
data = pd.read_csv(r"\\192.168.1.115\nas\MSData\PSMs_RAW.csv")

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
    dataset = SimpleDataset(data)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state = 42)
    test_dataset, validation_dataset = train_test_split(test_dataset, test_size=0.1, random_state = 42)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last = True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, drop_last = True)
    
    model = SimpleModel()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    SimpleModel.train_validate(model, train_loader, validation_loader, optimizer, criterion, 5)
    
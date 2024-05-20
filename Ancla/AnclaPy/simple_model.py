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
from sklearn.model_selection import learning_curve, train_test_split

#import csv as panda dataframe 
data = pd.read_csv(r"\\192.168.1.115\nas\MSData\PSMs_RAW.csv")
# data = pd.read_csv("/Volumes/NAS/MSData/PSMs_RAW.csv")
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
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt

    # Check if GPU is available and set TensorFlow to use it
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Load your data
    data = pd.read_csv(r"\\192.168.1.115\nas\MSData\PSMs_RAW.csv")

    # Extract peptide sequences and retention times from the DataFrame
    peptides = data["BaseSequence"].tolist()
    retention_times = data["ScanRetentionTime"].tolist()

    # Pad sequences to a fixed length
    max_length = 100
    padded_sequences = [seq.ljust(max_length, '0') for seq in peptides]

    # Define the dictionary mapping characters to indices
    char_to_index = {char: i + 1 for i, char in enumerate('ACDEFGHIKLMNPQRSTVWYUX')}
    # Add padding character '0' with index 0
    char_to_index['0'] = 0

    # Convert sequences to integers
    sequences_as_int = [[char_to_index[char] for char in seq] for seq in padded_sequences]

    # Convert to numpy arrays
    X_seq = np.array(sequences_as_int)
    y = np.array(retention_times)

    # Normalize the retention times
    scaler = MinMaxScaler((-1, 1))
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Train-test split
    X_seq_train, X_seq_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)

    # Define the neural network model
    model = Sequential([
        Embedding(input_dim=len(char_to_index), output_dim=128, input_length=max_length),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    
    #import keras
    import keras

    lr_scheldule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=30000,
            decay_rate=0.9)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate = lr_scheldule), loss='mean_squared_error')

    # Display the model summary
    print(model.summary())

    # Define EarlyStopping callback
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    history = model.fit(X_seq_train, y_train, epochs=50, batch_size=64,
                        validation_split=0.2, callbacks=[early_stopping])

    # Predict and scatter plot
    y_pred = model.predict(X_seq_test)
    
    #save model
    # model.save("model.h5")

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test, label='True Values', color='blue', s=0.3)
    plt.scatter(y_pred, y_test, label='Predictions', color='red', s=0.3)
    plt.xlabel('Predicted retention time')
    plt.ylabel('True retention time')
    plt.legend()
    # R2 score
    r2 = r2_score(y_test, y_pred)
    plt.text(0.5, 0.5, f'R2 score: {r2}', fontsize=12)
    plt.show()



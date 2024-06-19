import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from Ancla import Featurizer, RTDataset, BuModel, LandscapeExplorer, BottomUpResNet
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class SimpleLSTM(nn.Module):
    def __init__(self):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=100, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

        self.double()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":    
    # model = BottomUpResNet(num_blocks = 3)
    model = SimpleLSTM()

    vocab = pd.read_csv(r"D:\OtherPeptideResultsForTraining\vocab.csv")

    # make vocab a dictionary
    vocab = dict(zip(vocab["Id"], vocab["Token"]))
    #swap keys and values
    vocab = {v: k for k, v in vocab.items()}

    training_data = pd.read_csv(r"D:\DB_FullSequencesDistinct_noJurkat.csv")

    X, y = Featurizer.featurize_all_full_sequences(training_data, vocab, 100)

    X = Featurizer.pca_features(X)[:][0]
    
    y = Featurizer.normalize_targets(y)

    #remove low variance features
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()
    #remove features with zero variance and reomve the same ones in the y 
    X = selector.fit_transform(X.squeeze())
    y = np.delete(y, selector.get_support(indices = False))

    # Divide training features into train and test
    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    X_validate, X_test, y_validate, y_test = train_test_split(X_validate, y_validate, test_size=0.5, shuffle=True, random_state=42)
    
    # Create data sets
    train_dataset = RTDataset(X_train, y_train)
    validation_dataset = RTDataset(X_validate, y_validate)
    test_dataset = RTDataset(X_test, y_test)

    # Create data loaders
    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    criterion = torch.nn.L1Loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Explore the landscape 
    explorer = LandscapeExplorer(model = model, criterion = criterion, optimizer = optimizer,
                                    training_dataloader = train_loader,
                                    validation_dataloader = validation_loader,
                                    testing_dataset = test_dataset,
                                    num_points = 10, range_ = 1)
    
    explorer.train(epochs = 15)
    explorer.test()
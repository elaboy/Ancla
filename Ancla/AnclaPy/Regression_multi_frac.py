from Ancla import HelperFunctions, Visualize, RegressiveModels, TransformationFunctions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Ancla import Featurizer
import torch 
from torch.utils.data import Dataset, DataLoader
from torch.functional import F

batch_size = 32

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = torch.nn.Conv2d(batch_size, 32, 3, 1, 1)
        # self.pool = torch.nn.MaxPool2d(2, 2)
        self.cnn2 = torch.nn.Conv2d(batch_size, 32, 3, 1, 1)
        # self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(8 * 100, 1)

        self.double()
        
    def forward(self, x):
        x = F.relu(self.cnn(x))
        # x = self.pool(x)
        x = F.relu(self.cnn2(x))
        # x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x
    
class RTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
if __name__ == "__main__":

    # read all 10 fractions 
    frac1 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac1-calib-averaged.csv")
    frac2 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac2-calib-averaged.csv")
    frac3 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac3-calib-averaged.csv")
    frac4 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac4-calib-averaged.csv")
    frac5 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac5-calib-averaged.csv")
    frac6 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac6-calib-averaged.csv")
    frac7 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac7-calib-averaged.csv")
    frac8 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac8-calib-averaged.csv")
    frac9 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac9-calib-averaged.csv")
    frac10 = pd.read_csv(r"D:\OtherPeptideResultsForTraining\transformedData_12-18-17_frac10-calib-averaged.csv")

    training_data = pd.read_csv(r"D:\OtherPeptideResultsForTraining\fractionOverlapJurkatFromMannTryptic.csv")


    X = training_data["BaseSequence"].tolist()
    y = training_data["ScanRetentionTime"].tolist()

    training_features = Featurizer.featurize_all_normalized(X)

    training_dataset = RTDataset(training_features, y)

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    model = Model()

    criterion = torch.nn.MSELoss()

    #train the model 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(25):
        for i, data in enumerate(training_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            # print every 1000 mini-batches
            if i % 1000 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")


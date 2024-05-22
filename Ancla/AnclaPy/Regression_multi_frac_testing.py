from Ancla import HelperFunctions, Visualize, RegressiveModels, TransformationFunctions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Ancla import Featurizer
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.functional import F

batch_size = 64

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8 * 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(8 * 4)
        
        self.conv3 = nn.Conv1d(in_channels=16*2, out_channels=8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(8)
        
        # Calculate the size of the flattened features after the last pooling layer
        self._to_linear = None
        self._get_to_linear_dim()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.4)

        self.double()

        #initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _get_to_linear_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 8, 100)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            self._to_linear = x.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, self._to_linear)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
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
    
    # load the model 
    model = Model()
    model.load_state_dict(torch.load(r"D:\OtherPeptideResultsForTraining\RT_model_5_22_24.pth"))
    # Test model with predicting the fractions and making 10 plots with R^2

    # datasets
    fraction_datasets = []
    fractions_overlaps = []

    for fraction in [frac1, frac2, frac3, frac4, frac5, frac6, frac7, frac8, frac9, frac10]:
        
        X = fraction["BaseSequence"].tolist()
        y = fraction["Experimental"].tolist()

        #search in the training data for the X sequences and get the retention time
        overlaps = []
        for sequence in X:
            if sequence in training_data["BaseSequence"].tolist():
                overlaps.append(
                    training_data["ScanRetentionTime"].tolist()[training_data["BaseSequence"].tolist().index(sequence)])

        fraction_features = Featurizer.featurize_all_normalized(X)

        y = Featurizer.normalize_targets(y)

        fraction_dataset = RTDataset(fraction_features, y)

        fraction_datasets.append(fraction_dataset)
        fractions_overlaps.append(overlaps)
    
    model.eval()

    predictions = []
    for fraction in fraction_datasets:
        predictions.append(model(fraction.X))

    # plot the results and calculate R^2
    from sklearn.metrics import r2_score

    r2_scores = []

    # use the feature to search in the training data for the actual RT and then compare with the prediction
    for i, fraction in enumerate(fraction_datasets):
        actual_RT = fractions_overlaps[i]
        predicted_RT = predictions[i].detach().numpy()

        r2 = r2_score(actual_RT, predicted_RT)
        r2_scores.append(r2)

        plt.figure()
        plt.scatter(predicted_RT, actual_RT)
        plt.ylabel("Actual RT")
        plt.xlabel("Predicted RT")
        plt.title(f"Fraction {i+1}, R^2: {r2}")
        plt.savefig(f"fraction_{i+1}_5_22_24.png")
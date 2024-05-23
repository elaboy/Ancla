from Ancla import HelperFunctions, Visualize, RegressiveModels, TransformationFunctions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Ancla import Featurizer
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.functional import F

batch_size = 32

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(8 * 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(8 * 2)
        
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(8)
        
        # Calculate the size of the flattened features after the last pooling layer
        self._to_linear = None
        self._get_to_linear_dim()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.2)

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
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        
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

    training_features = Featurizer.featurize_all(X)

    y = Featurizer.normalize_targets(y)

    #divide training features into train and test
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(training_features, y, test_size=0.1, random_state=42)

    # Create data sets
    train_dataset = RTDataset(X_train, y_train)
    test_dataset = RTDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model()

    criterion = torch.nn.MSELoss()

    #train the model 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    model.to(device)

    # training loop
    for epoch in range(150):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{150}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    # Save the model
    torch.save(model.state_dict(), r"D:\OtherPeptideResultsForTraining\RT_model_5_22_24_V5_Adam_150Epochs_lr_001.pth")
    
    # Plot training history
    import matplotlib.pyplot as plt

    plt.plot(train_losses, label='Train Loss', color='r')
    plt.plot(val_losses, label='Validation Loss', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("training_history_5_22_24.png")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from Ancla import Featurizer, RTDataset, BuModel, ModelToolKit, LandscapeExplorer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

model = BuModel()

training_data = pd.read_csv(r"D:\OtherPeptideResultsForTraining\NO_JURKAT_fractionOverlapJurkatFromMannTryptic.csv")

X = training_data["BaseSequence"].tolist()
y = training_data["ScanRetentionTime"].tolist()

training_features = Featurizer.featurize_all(X)

y = Featurizer.normalize_targets(y)

# Divide training features into train and test
X_train, X_test, y_train, y_test = train_test_split(training_features, y, test_size=0.2, shuffle=True, random_state=42)

# Create data sets
train_dataset = RTDataset(X_train, y_train)
test_dataset = RTDataset(X_test, y_test)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)

# Explore the landscape 
explorer = LandscapeExplorer(model = model, criterion = criterion, optimizer = optimizer,
                              training_dataloader = train_loader, testing_dataset = test_dataset, 
                              epochs = 100, num_points = 300, range_ = 1)
explorer.run()
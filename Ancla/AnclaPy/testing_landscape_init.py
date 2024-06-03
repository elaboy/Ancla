import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from Ancla import Featurizer, RTDataset, BuModel, ModelToolKit
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

model = BuModel()

training_data = pd.read_csv(r"D:\OtherPeptideResultsForTraining\NO_JURKAT_fractionOverlapJurkatFromMannTryptic.csv")

X = training_data["BaseSequence"].tolist()
y = training_data["ScanRetentionTime"].tolist()

training_features = Featurizer.featurize_all(X)

y = Featurizer.normalize_targets(y)

#divide training features into train and test

X_train, X_test, y_train, y_test = train_test_split(training_features, y, test_size=0.2,
                                                        shuffle = True, random_state=42)

# Create data sets
train_dataset = RTDataset(X_train, y_train)
test_dataset = RTDataset(X_test, y_test)

training_example = train_dataset#[range(100)]
testing_example = test_dataset#[range(20)]

criterion = torch.nn.MSELoss() #maybe huberloss is better for this task

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9, weight_decay=1e-5)

# Function to train the model for a few steps
def train_step(model, X, y, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    outputs = model(X[0].reshape(1, 1, 7, 100))
    loss = criterion(outputs.reshape(-1), torch.from_numpy(y[1]).reshape(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# Train the model for a few steps
initial_params = [param.clone() for param in model.parameters()]
for _ in range(10):
    train_step(model, training_example[0], training_example[1], criterion, optimizer)
final_params = [param.clone() for param in model.parameters()]

# Function to generate random directions
def random_directions(model):
    direction_1 = [torch.randn_like(param) for param in model.parameters()]
    direction_2 = [torch.randn_like(param) for param in model.parameters()]
    return direction_1, direction_2

# Function to perturb model parameters
def perturb_params(params, directions, alpha, beta):
    perturbed_params = []
    for param, dir1, dir2 in zip(params, directions[0], directions[1]):
        perturbed_params.append(param + alpha * dir1 + beta * dir2)
    return perturbed_params

# Function to set model parameters
def set_model_params(model, params):
    with torch.no_grad():
        for param, param_new in zip(model.parameters(), params):
            param.copy_(param_new)

# Generate loss landscape
alphas = np.linspace(-0.5, 0.5, 30)
betas = np.linspace(-0.5, 0.5, 30)
losses = np.zeros((len(alphas), len(betas)))

original_params = [param.clone() for param in model.parameters()]
directions = random_directions(model)

for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        perturbed_params = perturb_params(original_params, directions, alpha, beta)
        set_model_params(model, perturbed_params)
        outputs = model(testing_example[0][0].reshape(1, 1, 7, 100))
        loss = criterion(outputs.reshape(-1), torch.from_numpy(testing_example[0][1]).reshape(-1))
        losses[i, j] = loss.item()

# Plot the loss landscape
X_mesh, Y_mesh = np.meshgrid(alphas, betas)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_mesh, Y_mesh, losses, cmap='viridis')
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('Loss')
ax.view_init(30, 120)  # <-- Adjust the view angle here
ax.invert_zaxis()  # Invert the z-axis if necessary
#make background less intense white to make it easier to see the plot
ax.set_facecolor('lightgrey')
plt.show()
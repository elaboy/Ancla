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

# Divide training features into train and test
X_train, X_test, y_train, y_test = train_test_split(training_features, y, test_size=0.2, shuffle=True, random_state=42)

# Create data sets
train_dataset = RTDataset(X_train, y_train)
test_dataset = RTDataset(X_test, y_test)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)

model.to("cuda")

# Function to train the model for a few steps
def train_step(model, X, y, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    outputs = model(X.reshape(len(X), 1, 7, 100).to("cuda"))
    loss = criterion(outputs.reshape(-1).to("cpu"), y.reshape(-1).to("cpu"))
    loss.backward()
    optimizer.step()
    return loss.item()

# Train the model for a few steps
initial_params = [param.clone() for param in model.parameters()]
for _ in range(10):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        train_step(model, inputs, targets, criterion, optimizer)
final_params = [param.clone() for param in model.parameters()]

# Function to generate random directions
def random_directions(model):
    direction_1 = [torch.randn_like(param) for param in model.parameters()]
    direction_2 = [torch.randn_like(param) for param in model.parameters()]
    return direction_1, direction_2

# Function to interpolate between parameters
def interpolate_params(initial_params, final_params, alpha, beta, direction_1, direction_2):
    interpolated_params = []
    for param_init, param_final, dir1, dir2 in zip(initial_params, final_params, direction_1, direction_2):
        interpolated_params.append(param_init + alpha * (param_final - param_init) + beta * dir2)
    return interpolated_params

# Function to set model parameters
def set_model_params(model, params):
    with torch.no_grad():
        for param, param_new in zip(model.parameters(), params):
            param.copy_(param_new)

# Generate a few test points to determine the highest loss
alphas = np.linspace(-1, 1, 5)  # Fewer points for initial test
betas = np.linspace(-1, 1, 5)
highest_loss = 0

original_params = [param.clone() for param in model.parameters()]
direction_1, direction_2 = random_directions(model)

for alpha in alphas:
    for beta in betas:
        interpolated_params = interpolate_params(original_params, final_params, alpha, beta, direction_1, direction_2)
        set_model_params(model, interpolated_params)
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs.flatten(), targets.flatten())
                total_loss += loss.item()
            highest_loss = max(highest_loss, total_loss / len(train_loader))
            print(f'testing points | Alpha: {alpha}, Beta: {beta}, Loss: {total_loss}')

# Generate loss landscape by interpolating in two directions
alphas = np.linspace(-1, 1, 40)
betas = np.linspace(-1, 1, 40)
losses = np.zeros((len(alphas), len(betas)), dtype=np.float64)  # Initialize losses array

path_alphas = []
path_betas = []
path_losses = []

for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        # Interpolate parameters
        interpolated_params = interpolate_params(original_params, final_params, alpha, beta, direction_1, direction_2)
        set_model_params(model, interpolated_params)

        # Compute loss for all test examples
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to("cuda"), targets.to("cuda")
                outputs = model(inputs)
                loss = criterion(outputs.flatten(), targets.flatten())
                total_loss += loss.item()

        # Average loss across all test examples and take the logarithm
        average_loss = total_loss / len(test_loader)
        log_loss = np.log(average_loss + 1e-10)  # Adding a small value to avoid taking the log of zero
        losses[i, j] = log_loss
        print(f'landscape test | Alpha: {alpha}, Beta: {beta}, Loss: {log_loss}')

        # Record path for the line plot
        if alpha in alphas and beta in betas:
            path_alphas.append(alpha)
            path_betas.append(beta)
            path_losses.append(log_loss)

# Plot the contour plot of the loss landscape
plt.contourf(alphas, betas, losses, cmap='viridis')
plt.xlabel('Alpha')
plt.ylabel('Beta')
plt.colorbar(label='Log Loss')
plt.title('Contour Plot of Log Loss Landscape')

# Mesh plot
fig = plt.figure()
X_, Y_ = np.meshgrid(alphas, betas)
ax = fig.add_subplot(111, projection='3d')
ax.contourf(X_, Y_, losses, cmap='viridis', zdir='z', offset=losses.min())
ax.set_xlabel('Alpha')
ax.set_ylabel('Beta')
ax.set_zlabel('Log Loss')
ax.set_title('Contour Plot of Log Loss Landscape')

# Plot the mesh
ax.plot_surface(X_, Y_, losses, cmap='viridis', alpha=0.5)

# Plot the line representing the movement
ax.plot(path_alphas, path_betas, path_losses, marker='o', color='r', linestyle='-')

plt.show()
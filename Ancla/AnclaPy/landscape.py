import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimpleFCC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFCC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create random dataset
X = torch.rand(10, 5)
y = torch.rand(10, 2)

# Create model
model = SimpleFCC(5, 10, 2)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model for a few epochs
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}: Loss {loss.item()}')

# Generate random directions
direction1 = [torch.randn_like(p) for p in model.parameters()]
direction2 = [torch.randn_like(p) for p in model.parameters()]

# Function to compute loss
def get_loss(model, criterion, X, y):
    model.eval()
    with torch.no_grad():
        output = model(X)
        print(f"Output shape: {output.shape}, Target shape: {y.shape}")  # Debugging line
        loss = criterion(output, y)
    model.train()
    return loss.item()

# Function to compute the loss landscape
def loss_landscape(model, criterion, X, y, direction1, direction2, num_points=10000, range_=5.0):
    original_params = [p.clone() for p in model.parameters()]
    losses = np.zeros((num_points, num_points))
    x_grid = np.linspace(-range_, range_, num_points)
    y_grid = np.linspace(-range_, range_, num_points)

    for i, xi in enumerate(x_grid):
        for j, yj in enumerate(y_grid):
            for k, p in enumerate(model.parameters()):
                p.data = original_params[k] + xi * direction1[k] + yj * direction2[k]
            losses[i, j] = get_loss(model, criterion, X, y)

    # Restore original parameters
    for k, p in enumerate(model.parameters()):
        p.data = original_params[k]
    
    return x_grid, y_grid, losses

# Compute loss landscape
x_grid, y_grid, losses = loss_landscape(model, criterion, X, y, direction1, direction2, num_points=50, range_=1.0)

# Plot the loss landscape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X_, Y_ = np.meshgrid(x_grid, y_grid)
ax.plot_surface(X_, Y_, losses, cmap='viridis')

ax.set_xlabel('Direction 1')
ax.set_ylabel('Direction 2')
ax.set_zlabel('Loss')
plt.title('Loss Landscape')
plt.show()

# Contour plot
plt.contour(X_, Y_, losses, levels=50, cmap='viridis')
plt.xlabel('Direction 1')
plt.ylabel('Direction 2')
plt.title('Contour Plot of Loss Landscape')
plt.colorbar()
plt.show()

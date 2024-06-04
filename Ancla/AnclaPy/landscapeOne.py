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

batch_size = 128
trained = False
    
if __name__ == "__main__":

    # read all 10 fractions 
    frac1 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac1-calib-averaged.csv")
    frac2 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac2-calib-averaged.csv")
    frac3 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac3-calib-averaged.csv")
    frac4 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac4-calib-averaged.csv")
    frac5 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac5-calib-averaged.csv")
    frac6 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac6-calib-averaged.csv")
    frac7 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac7-calib-averaged.csv")
    frac8 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac8-calib-averaged.csv")
    frac9 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac9-calib-averaged.csv")
    frac10 = pd.read_csv(r"D:\no_jurkat_transformedData_12-18-17_frac10-calib-averaged.csv")

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

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BuModel()

    model.to(device)

    criterion = torch.nn.MSELoss() #maybe huberloss is better for this task

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9, weight_decay=1e-5)

    train_losses = []
    val_losses = []

    epochs = 1

    landscape_subsample = train_dataset.__getitem__(100)
    # take 100 freom the landscape subsample
    # landscape_subsample = RTDataset(landscape_subsample[0][:100], landscape_subsample[1][:100])
    if trained == False:
        model.train()
        running_loss = 0.0
        
        direction1 = [torch.randn_like(p) for p in model.parameters()]
        direction2 = [torch.randn_like(p) for p in model.parameters()]
        
        ModelToolKit.landscape(model = model, criterion = criterion, 
                               X = landscape_subsample[0], y = landscape_subsample[1],
                               direction1 = direction1, direction2 = direction2,
                               num_points = 300, range_ = 10.0, device = device)
        
        #     model.eval()
        #     val_loss = 0.0
        #     with torch.no_grad():
        #         for inputs, targets in test_loader:
        #             outputs = model(inputs.to(device))
        #             loss = criterion(outputs, targets.reshape(-1).to(device))
        #             val_loss += loss.item()
        #             # break

        #     val_loss /= len(test_loader)
        #     val_losses.append(val_loss)
            
        #     # print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        
        # # save model
        # model.eval()
        # torch.save(model.state_dict(), "model_LandscapeTest2.pt")
        # model.train()

    else: 
        model.load_state_dict(torch.load(r"C:\Users\elabo\Documents\GitHub\Ancla\Ancla\model_LandscapeTest1.pt"))

# direction1 = [torch.randn_like(p) for p in model.parameters()]
# direction2 = [torch.randn_like(p) for p in model.parameters()]


# # Compute loss landscape
# x_grid, y_grid, losses = ModelToolKit.loss_landscape(model, criterion,
#                                                       test_dataset.__getitem__(range(len(test_dataset)//4))[0].reshape(len(range(len(test_dataset)//4)),
#                                                                                                                     1, 7, 100),
#                                                       test_dataset.__getitem__(range(len(test_dataset)//4))[1],
#                                                       direction1, direction2, num_points=5, range_=30.0, device = device)

# # Plot the loss landscape
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X_, Y_ = np.meshgrid(x_grid, y_grid)
# ax.plot_surface(X_, Y_, losses, cmap='viridis')

# ax.set_xlabel('Direction 1')
# ax.set_ylabel('Direction 2')
# ax.set_zlabel('Loss')
# plt.title('Loss Landscape')
# plt.show()

# # Contour plot
# plt.contour(X_, Y_, losses, levels=50, cmap='viridis')
# plt.xlabel('Direction 1')
# plt.ylabel('Direction 2')
# plt.title('Contour Plot of Loss Landscape')
# plt.colorbar()
# plt.show()
from Ancla import BuModel, RTDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Ancla import Featurizer
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.functional import F

batch_size = 5
    
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
    from sklearn.model_selection import train_test_split

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

    #print the model architecture 
    print(model)

    criterion = torch.nn.MSELoss() #maybe huberloss is better for this task

    #train the model 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum = 0.9, weight_decay=1e-5)

    # RecudeLRonPlateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=0.1, patience=10)

    train_losses = []
    val_losses = []

    model.to(device)

    #print both training and testing dataloader size 
    print(len(train_loader), len(test_loader))


    # training loop
    for epoch in range(30):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.reshape(-1, 1).to(device))
            loss.backward()

            #gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs.to(device))
                loss = criterion(outputs, targets.reshape(-1, 1).to(device))
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{30}, Train Loss: {train_loss}, Validation Loss: {val_loss}")
        scheduler.step(val_loss)

    # Save the model
    torch.save(model.state_dict(),
                r"D:\OtherPeptideResultsForTraining\RT_model_5_24_2024_V9_NO_JURKAT_TRAIN.pth")
    
    # Plot training history
    import matplotlib.pyplot as plt

    plt.plot(train_losses, label='Train Loss', color='r')
    plt.plot(val_losses, label='Validation Loss', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("training_history_5_24_2024_V9.png")
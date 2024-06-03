import stat
from statistics import LinearRegression
from numpy._typing import NDArray
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import peptides
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils
from torch.utils.data import Dataset, DataLoader

aa_dict = {
    "PAD" : 0,
    "A" : 1,
    "R" : 2,
    "N" : 3,
    "D" : 4,
    "C" : 5,
    "E" : 6,
    "Q" : 7,
    "G" : 8,
    "H" : 9,
    "I" : 10,
    "L" : 11,
    "K" : 12,
    "M" : 13,
    "F" : 14,
    "P" : 15,
    "S" : 16,
    "T" : 17,
    "W" : 18,
    "Y" : 19,
    "V" : 20,
    "U" : 21,
}

class Featurizer(object):
    @staticmethod
    def featurize(data: str) -> NDArray:
        # the array will be of shape (peptide, residues, features)

        # split the string into a list of characters
        peptides_as_char_list = [list(data)]

        # add PAD until the length is 100
        padded_peptides_char_list = [peptide + ["PAD"] *
                                      (100 - len(peptide)) for peptide in peptides_as_char_list]

        # conver data into a list of peptides where each character is replaces by its corresponding value in the aa_dict
        peptides_as_dict_index = [[aa_dict[aa] for aa in peptide] 
                                  for peptide in padded_peptides_char_list]
        
        # compute data features per residue 
        hydrophobicity = [peptides.Peptide(residue).hydrophobic_moment() for residue in data]
        
        #pad the hydrophobicity array with zeros until the length is 100
        hydrophobicity = [hydrophobicity + [0] * (100 - len(hydrophobicity))]
        
        # compute the z-scales
        z_scales = [peptides.Peptide(residue).z_scales() for residue in data]
        
        #pad the z-scales array with zeros until the length is 100
        z_scales = [z_scales + [0] * (100 - len(z_scales))]

        z1_scale, z2_scale, z3_scale, z4_scale, z5_scale = [], [], [], [], []

        for z_scale in z_scales:
            for i, z in enumerate(z_scale):
                if z != 0:
                    z1_scale.append(z.z1)
                    z2_scale.append(z.z2)
                    z3_scale.append(z.z3)
                    z4_scale.append(z.z4)
                    z5_scale.append(z.z5)
                else:
                    z1_scale.append(0)
                    z2_scale.append(0)
                    z3_scale.append(0)
                    z4_scale.append(0)
                    z5_scale.append(0)

        # compute the charge
        # charge = [peptides.Peptide(residue).charge() for residue in data]

        #pad the charge array with zeros until the length is 100
        # charge = [charge + [0] * (100 - len(charge))]

        # add the features into the final array, vstacked
        final_array = np.vstack(
            [np.array(peptides_as_dict_index), 
             np.array(hydrophobicity),
             np.array(z1_scale),
             np.array(z2_scale),
             np.array(z3_scale),
             np.array(z4_scale),
             np.array(z5_scale), 
            #  np.array(charge)]
            ])

        return final_array
    
    @staticmethod
    def featurize_all(data: list) -> NDArray:
        # parallelize the featurization of the data
        from joblib import Parallel, delayed

        # featurize the data
        featurized_data = Parallel(n_jobs=-1)(delayed(Featurizer.featurize)(data) for data in data)

        return np.stack(featurized_data)
    
    @staticmethod
    def normalize_targets(targets: list) -> NDArray:
        from sklearn.preprocessing import StandardScaler
        
        #create the scaler
        scaler = StandardScaler()
        
        # make it a (targets_length, 1)
        targets = np.array(targets).reshape(-1, 1)

        #fit the scaler and transform the columns Database, Experimental and PostTransformation
        targets = scaler.fit_transform(targets)
        
        return targets
        
    @staticmethod
    def min_max_scaler(data: NDArray) -> NDArray:
        # Assuming 'data' is your 3D array
        mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
        std_dev = np.std(data, axis=(0, 1, 2), keepdims=True)

        # Avoid division by zero
        std_dev = np.where(std_dev == 0, 1e-9, std_dev)

        # Apply feature-wise normalization
        normalized_data = (data - mean) / std_dev

        return normalized_data

    @staticmethod
    def featurize_all_normalized(data: list) -> NDArray:
        return Featurizer.min_max_scaler(Featurizer.featurize_all(data))

class HelperFunctions(object):
    
    #read data from csv. Headers are FullSequence, Database, Experimental, PostTransformation
    @staticmethod
    def read_results(path: str) -> pd.DataFrame:
        data = pd.read_csv(path)
        return data
    
    @staticmethod
    def save_results(data: pd.DataFrame, path: str) -> None:
        data.to_csv(path, index=False)

    @staticmethod
    def standardScaler(data: pd.DataFrame) -> pd.DataFrame:
        from sklearn.preprocessing import StandardScaler
        
        #create the scaler
        scaler = StandardScaler()
        
        #fit the scaler and transform the columns Database, Experimental and PostTransformation
        data[['Database', 'Experimental', 'PostTransformation']] = scaler.fit_transform(
            data[['Database', 'Experimental', 'PostTransformation']])

    @staticmethod
    def reject_outliers(data: pd.DataFrame) -> pd.DataFrame:
        from scipy.stats import zscore
        
        # reject outliers from the Database and Experimental columns
        data['Database_z'] = zscore(data['Database'])
        data['Experimental_z'] = zscore(data['Experimental'])

        #reject the outliers
        data = data[(np.abs(data['Database_z']) < 3) & (np.abs(data['Experimental_z']) < 3)]

        #drop the z-score columns
        data = data.drop(columns=['Database_z', 'Experimental_z'])

        return data
        
class Visualize(object):
    
    @staticmethod
    def scatter_plots(data: pd.DataFrame) -> None:
        """ Return None, only shows the plots.
        
        Three scatter plots: Database vs Database, Experimental vs Database, PostTransformation vs Database
        """
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
        from sklearn.metrics import r2_score

        #first plot
        axs[0].scatter(data['Database'], data['Database'], 
                       color='red', label='Database vs Database', s=0.3)
        axs[0].scatter(data['Experimental'], data['Database'], 
                       color='blue', label='Experimental vs Database', s=0.3)
        axs[0].set_xlabel('Experimental Peptide RT')
        axs[0].set_ylabel('Database Peptide RT')
        axs[0].legend()
        
        #second plot
        axs[1].scatter(data['Database'], data['Database'],
                       color='red', label='Database vs Database', s=0.3)
        # axs[1].scatter(data['Experimental'], data['Database'],
        #                color='blue', label='Experimental vs Database', s=0.2)
        axs[1].scatter(data['PostTransformation'], data['Database'],
                       color='orange', label='PostTransformation vs Database', s=0.3)
        axs[1].set_xlabel('PostTransformation')
        axs[1].set_ylabel('Database')
        axs[1].legend()
    
        #dd R2 score to the plot
        r2_exp = r2_score(data['Database'], data['Experimental'])
        axs[0].text(0.5, 0.9, f'R2: {r2_exp:.2f}', transform=axs[0].transAxes)

        r2 = r2_score(data['Database'], data['PostTransformation'])
        axs[1].text(0.5, 0.9, f'R2: {r2:.2f}', transform=axs[1].transAxes)
        
    @staticmethod
    def z_scores_hist(data: pd.DataFrame) -> None:
        from scipy.stats import zscore
        
        #compute the z-scores for the Database and Experimental columns, then database and PostTransformation
        data['Database_z'] = zscore(data['Database'])
        data['Experimental_z'] = zscore(data['Experimental'])
        data['PostTransformation_z'] = zscore(data['PostTransformation'])
        
        #show the z-scores before and after the transformation. I want to see if the distribution is the same
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        axs[0].hist(data['Database_z'], bins=100, color='red', alpha=0.5, label='Database')
        axs[0].hist(data['Experimental_z'], bins=100, color='blue', alpha=0.5, label='Experimental')
        axs[0].set_title('Before transformation')
        axs[0].set_xlabel('Z-score')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        
        axs[1].hist(data['Database_z'], bins=100, color='red', alpha=0.5, label='Database')
        axs[1].hist(data['PostTransformation_z'], bins=100, color='orange', alpha=0.5, label='PostTransformation')
        axs[1].set_title('After transformation')
        axs[1].set_xlabel('Z-score')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        
        #add trendline to the histograms
        from scipy.stats import norm
        from scipy import stats
       
        #fit a normal distribution to the data
        mu, std = norm.fit(data['Database_z'])
        x = np.linspace(-5, 5, 100)
        p = norm.pdf(x, mu, std)
        
        axs[0].plot(x, p, 'k', linewidth=2)
        axs[1].plot(x, p, 'k', linewidth=2)

    @staticmethod
    def residuals_scatter(data: pd.DataFrame) -> None:
        #compute the residuals for the experimental and post transformation columns
        data['Residuals_exp'] = data['Database'] - data['Experimental']
        data["Residuals_trans"] = data['Database'] - data['PostTransformation']
        
        #plot the data and the residuals in two plots in the same grid 
        fig, axs = plt.subplots(2, 2, figsize=(15, 5))
        
        axs[0, 0].scatter(data['Database'], data['Database'],
                          color='red', label='Database vs Database', s=0.5)
        axs[0, 0].scatter(data['Experimental'], data['Database'],
                          color='blue', label='Experimental vs Database', s=0.5)
        axs[0, 0].set_xlabel('Experimental')
        axs[0, 0].set_ylabel('Database')
        axs[0, 0].legend()
        
        axs[0, 1].scatter(range(len(data["Residuals_exp"])), data['Residuals_exp'], 
                          color='blue', label='Residuals Experimental', s=0.5)
        #horizontal black line at 0
        axs[0, 1].axhline(0, color='black', lw=1)
        axs[0, 1].set_xlabel('Residuals Experimental')
        axs[0, 1].set_ylabel('Database')
        axs[0, 1].legend()
        
        axs[1, 0].scatter(data['Database'], data['Database'],
                          color='red', label='Database vs Database', s=0.5)
        axs[1, 0].scatter(data['PostTransformation'], data['Database'],
                          color='green', label='PostTransformation vs Database', s=0.5)
        axs[1, 0].set_xlabel('PostTransformation')
        axs[1, 0].set_ylabel('Database')
        axs[1, 0].legend()
        
        axs[1, 1].scatter(range(len(data["Residuals_trans"])), data["Residuals_trans"],
                          color='green', label='Residuals PostTransformation', s=0.5)
        #horizontal black line at 0
        axs[1, 1].axhline(0, color='black', lw=1)
        axs[1, 1].set_xlabel('Residuals PostTransformation')
        axs[1, 1].set_ylabel('Database')
        axs[1, 1].legend()
        

    @staticmethod
    def qq_plot(data: pd.DataFrame) -> None:
        from scipy.stats import probplot
        
        #compute the residuals for the experimental and post transformation columns
        data['Residuals_exp'] = data['Database'] - data['Experimental']
        data["Residuals_trans"] = data['Database'] - data['PostTransformation']
        
        #plot the qq plot for the residuals
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        probplot(data['Residuals_exp'], plot=axs[0])
        axs[0].set_title('QQ plot for the residuals of the Experimental column')
        
        probplot(data['Residuals_trans'], plot=axs[1])
        axs[1].set_title('QQ plot for the residuals of the PostTransformation column')

    @staticmethod
    def visualize_all(data: pd.DataFrame) -> None:
        Visualize.scatter_plots(data)
        Visualize.z_scores_hist(data)
        # Visualize.residuals_scatter(data)
        # Visualize.qq_plot(data)
        plt.show()

    @staticmethod
    def visualize_scatter_distributions(data: pd.DataFrame) -> None:
        from sklearn.metrics import r2_score
        # 4 by 4 grid. top two scatter, bottom two histograms
        fig, axs = plt.subplots(2, 2, figsize=(24, 16))

        #plot the scatter plots
        axs[0, 0].scatter(data['Database'], data['Database'],
                            color='red', label='Database vs Database', s=0.5)
        axs[0, 0].scatter(data['Experimental'], data['Database'],
                            color='blue', label='Experimental vs Database', s=0.5)
        axs[0, 0].set_xlabel('Experimental')
        axs[0, 0].set_ylabel('Database')
        #r2 scores
        r2_exp = r2_score(data['Database'], data['Experimental'])
        axs[0, 0].text(0.5, 0.5, f'R2: {r2_exp:.2f}', transform=axs[0, 0].transAxes)
        axs[0, 0].legend()

        axs[0, 1].scatter(data['Database'], data['Database'],
                            color='red', label='Database vs Database', s=0.5)
        axs[0, 1].scatter(data['PostTransformation'], data['Database'],
                            color='blue', label='PostTransformation vs Database', s=0.5)
        axs[0, 1].set_xlabel('PostTransformation')
        axs[0, 1].set_ylabel('Database')
        axs[0, 1].legend()
        
        # histograms
        axs[1, 0].hist(data['Database'], bins=100, color='red', alpha=0.5, label='Database')
        axs[1, 0].hist(data['Experimental'], bins=100, color='blue', alpha=0.5, label='Experimental')
        axs[1, 0].set_title('Database and Experimental')
        axs[1, 0].set_xlabel('RT')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].legend()

        axs[1, 1].hist(data['Database'], bins=100, color='red', alpha=0.5, label='Database')
        axs[1, 1].hist(data['PostTransformation'], bins=100, color = "blue", alpha=0.5, label='PostTransformation')
        axs[1, 1].set_title('Database and PostTransformation')
        axs[1, 1].set_xlabel('RT')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].legend()

        plt.show()

    @staticmethod
    def save_visualize_fractions(data: pd.DataFrame, path: str) -> None:
        from sklearn.metrics import r2_score

        Visualize.scatter_plots(data)
        plt.savefig(path + "/_scatter_plots.png")
        plt.clf()
        Visualize.z_scores_hist(data)
        plt.savefig(path + "/_z_scores_hist.png")
        plt.clf()
        Visualize.residuals_scatter(data)
        plt.savefig(path + "/_residuals_scatter.png")
        # Visualize.qq_plot(data).savefig(path + "_qq_plot.png")
        plt.clf()

class RegressiveModels(object):
    @staticmethod
    def fit_linear(data: pd.DataFrame, x: str="Experimental", y: str="Database") -> NDArray:
        from sklearn.linear_model import LinearRegression
    
        model = LinearRegression()
    
        #fit the model
        model.fit(data[x].values.reshape(-1, 1), data[y])
    
        return model

    @staticmethod
    def fit_lowess(data: pd.DataFrame, x: str="Experimental", y: str="Database", 
                  frac: float=1./3) -> NDArray:
    
        from statsmodels.nonparametric.smoothers_lowess import lowess
    
        # order the dataframe by the Database column
        data = data.sort_values(by=[y])

        #take the values of the columns
        database = data[y].to_list()
        experimental = data[x].to_list()
    
        #fit the lowess model
        model = lowess(database, experimental, frac = frac, it = 4)
    
        return model

class TransformationFunctions(object):
    @staticmethod    
    def transform_experimental_lowess(data: pd.DataFrame, lowess_model: NDArray,
                               x: str="Experimental", y: str="Database") -> pd.DataFrame:
        #transform the experimental values
        data['PostTransformation'] = np.interp(data['Experimental'], lowess_model[:, 0], lowess_model[:, 1])
    
        return data
    
    @staticmethod
    def transform_experimental_linear(data: pd.DataFrame, linear_model: LinearRegression,
                               x: str="Experimental", y: str="Database") -> pd.DataFrame:
        #transform the experimental values
        data['PostTransformation'] = linear_model.predict(data[x].values.reshape(-1, 1))
    
        return data

class BuModel(nn.Module):
    def __init__(self):
        super(BuModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(4)
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(8)
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(16)
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias = False)
        self.bn4 = nn.BatchNorm2d(32)
        # self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias = False)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Calculate the size of the flattened features after the last pooling layer
        self._to_linear = None
        self._get_to_linear_dim()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 4)
        self.fc7 = nn.Linear(4, 2)
        self.fc8 = nn.Linear(2, 1)
        
        self.dropout = nn.Dropout(0.5)

        self.double()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _get_to_linear_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 7, 100)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            self._to_linear = x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor):
        # make sure the input tensor is of shape (batch_size, 1, 7, 100)
        # x = x.view(batch_size, 1, 7, 100)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(-1, self._to_linear)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)

        return x
    
class RTDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        return torch.tensor(self.X[idx]).unsqueeze_(0), self.y[idx]
    
class ModelToolKit(object):

    # Function to compute loss
    @staticmethod
    def get_loss(model, criterion, X, y, device = "cpu"):
        model.eval()
        with torch.no_grad():
            # make X reshape to (batch_size, 1, 7, 100)
            X = X.reshape(-1, 1, 7, 100)
            output = model(X.to(device))
            # print(f"Output shape: {output.shape},\
            #     Target shape: {y.shape}")  # Debugging line
            loss = criterion(output.reshape(-1).to("cpu"), torch.from_numpy(y).reshape(-1).to("cpu"))
        model.train()
        return loss.item()

    # Function to compute the loss landscape
    @staticmethod
    def loss_landscape(model, criterion, X, y, direction1,
                        direction2, num_points=50, range_=5.0, device = "cpu"):
        original_params = [p.clone() for p in model.parameters()]
        losses = np.zeros((num_points, num_points))
        x_grid = np.linspace(-range_, range_, num_points)
        y_grid = np.linspace(-range_, range_, num_points)

        for i, xi in enumerate(x_grid):
            for j, yj in enumerate(y_grid):
                for k, p in enumerate(model.parameters()):
                    p.data = original_params[k] + xi * \
                            direction1[k] + yj * direction2[k]
                losses[i, j] = ModelToolKit.get_loss(model, criterion, X, y, device = device)
                print("Calculating Loss for Landscape: ", i, j)

        # Restore original parameters
        for k, p in enumerate(model.parameters()):
            p.data = original_params[k]
        
        return x_grid, y_grid, losses
    
    # Function to compute the loss landscape and plot it. Used for model evaluation when it is not trained.
    @staticmethod
    def landscape(model, criterion, X, y, direction1, direction2, num_points=50, range_=5.0, device = "cpu"):
        x_grid, y_grid, losses = ModelToolKit.loss_landscape(model, criterion, X, y,
                                                              direction1, direction2, num_points,
                                                                range_, device = device)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X_, Y_ = np.meshgrid(x_grid, y_grid)
        ax.plot_surface(X_, Y_, losses, cmap='viridis')
        plt.show()
    
    # Interactive visualization of the loss landscape during training
    @staticmethod
    def landscape_live(model, optimizer, criterion, epochs: int, data_loader: DataLoader, landscape_dataset: RTDataset,
                        direction1, direction2, num_points, range_, device = "cpu"):
        # Enable interactive mode
        plt.ion()

        fig = plt.figure(figsize=(14, 6))
        ax3d = fig.add_subplot(121, projection='3d')
        ax2d = fig.add_subplot(122)
        
        # Training loop with live visualization
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                loss = criterion(outputs.reshape(-1), targets.reshape(-1).to(device))
                loss.backward()
                optimizer.step()
                # print(f'Epoch {epoch}: Loss {loss.item()}')

            if epoch % 1 == 0:  # Visualize every epoch
                x_grid, y_grid, losses = ModelToolKit.loss_landscape(model, criterion,
                    landscape_dataset[0],
                    landscape_dataset[1],
                    direction1, direction2,
                    num_points, range_,
                    device = device)

                ax3d.cla()
                ax2d.cla()

                X_, Y_ = np.meshgrid(x_grid, y_grid)
                ax3d.plot_surface(X_, Y_, losses, cmap='viridis')
                ax3d.set_xlabel('Direction 1')
                ax3d.set_ylabel('Direction 2')
                ax3d.set_zlabel('Loss')
                ax3d.set_title(f'Loss Landscape at Epoch {epoch}')

                contour = ax2d.contour(X_, Y_, losses, levels=50, cmap='viridis')
                ax2d.set_xlabel('Direction 1')
                ax2d.set_ylabel('Direction 2')
                ax2d.set_title(f'Contour Plot of Loss Landscape at Epoch {epoch}')
                fig.colorbar(contour, ax=ax2d)

                plt.draw()
                plt.pause(0.15)

        # Turn off interactive mode
        plt.ioff()
        plt.show()
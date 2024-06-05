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
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
            #  np.array(hydrophobicity),
             np.array(z1_scale),
             np.array(z2_scale),
             np.array(z3_scale),
             np.array(z4_scale),
             np.array(z5_scale), 
            #  np.array(charge)]
            ])

        return final_array
    
    @staticmethod
    def featurize_all(data: list) -> list:
        #run with tqdm for a progress bar, in parallel
        with Pool(processes= cpu_count()) as pool:
            features = list(tqdm(pool.imap_unordered(Featurizer.featurize, data), total=len(data), desc='Featurizing'))

        # list within list to list
        features = [feature for feature in features]

        return features
    
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
    def __init__(self) -> torch.nn.Module:
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

        # # Initialize weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight)
        #         # nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)

    def _get_to_linear_dim(self) -> None:
        with torch.no_grad():
            x = torch.zeros(1, 1, 6, 100)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            self._to_linear = x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: nn.Module = None) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BottomUpResNet(nn.Module):
    def __init__(self, num_blocks: int) -> None:
        super(BottomUpResNet, self).__init__()
        self.in_channels = 4
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(32, num_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ltsm = nn.LSTM(32, 16, 1, batch_first=True)
        self.fc = nn.Linear(16, 1)

        self.double()

    def make_layer(self, out_channels: int, num_blocks: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = x.view(1, 32, 32)
        x, _ = self.ltsm(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
    
class RTDataset(Dataset):

    def __init__(self, X: NDArray, y: NDArray) -> None:
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> tuple:

        return torch.from_numpy(np.array(self.X[idx], dtype = np.float64)).unsqueeze_(0), self.y[idx]
    
class LandscapeExplorer():
    '''
    Class to explore the loss landscape of a model by interpolating between two points in two directions. 
        Paper: https://arxiv.org/abs/1712.09913
    '''
    def __init__(self, model: torch.nn, criterion : torch.nn, optimizer : torch.optim.Optimizer,
                 training_dataloader : DataLoader, validation_dataloader: DataLoader, testing_dataset: Dataset, num_points : int, range_ : float) -> None:
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device) #move to device
        self.optimizer = optimizer
        self.criterion = criterion
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.testing_dataset = testing_dataset
        self.direction_1, self.direction_2 = self.__random_directions__()
        self.num_points = num_points
        self.range_ = range_
        # self.initial_params = [param.clone() for param in self.model.parameters()]
        self.path_alphas = []
        self.path_betas = []
        self.path_losses = []
        self.trainin_losses = []
        self.testing_losses = []
        self.validation_losses = []

    # Function to train the model for a few steps
    def __train_step__(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(X.to(self.device))
        loss = self.criterion(outputs.reshape(-1), y.reshape(-1).to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    # Function to generate random directions
    def __random_directions__(self) -> list:
        direction_1 = [torch.randn_like(param) for param in self.model.parameters()]
        direction_2 = [torch.randn_like(param) for param in self.model.parameters()]
        return direction_1, direction_2
    
    # Function to set model parameters
    def __set_model_params__(self, params: list) -> None:
        with torch.no_grad():
            for param, param_new in zip(self.model.parameters(), params):
                param.copy_(param_new)



    # Function to interpolate between two points
    def __interpolate_params__(self, initial_params, final_params, alpha, beta) -> tuple:
        interpolated_params = []
        for param_init, param_final in zip(initial_params, final_params):
            dir1 = param_final - param_init
            interpolated_params.append(param_init + alpha * dir1 + beta * torch.randn_like(param_init))
        return interpolated_params
    
    def __calculate_loss__(self, alpha_beta) -> tuple:
        alpha, beta = alpha_beta
        interpolated_params = self.__interpolate_params__(self.initial_params, self.final_params, alpha, beta)
        self.__set_model_params__(interpolated_params)
        
        total_loss = 0
        with torch.no_grad():
            # inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(self.testing_dataset[0])
            loss = self.criterion(outputs.reshape(-1), self.testing_dataset[1].reshape(-1))
            total_loss += loss.item()
        
        average_loss = total_loss / len(self.testing_dataset)
        # log_loss = np.log(average_loss + 1e-10)  # Adding a small value to avoid taking the log of zero

        #empty gpu cache | VERY IMPORTANT
        # torch.cuda.empty_cache()

        return (alpha, beta, average_loss)

    def __prepare_testing_dataset__(self):
            self.testing_dataset = (self.testing_dataset[:][0].unsqueeze(1).reshape(self.testing_dataset[:][0].shape[1], 1, 6, 100).to(self.device),
                                     torch.from_numpy(self.testing_dataset[:][1]).to(self.device))

    def __validation__(self) -> None:
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for inputs, targets in tqdm(self.validation_dataloader, total = len(self.validation_dataloader), desc='Validating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs.reshape(-1), targets.reshape(-1))
                total_loss += loss.item()
            average_loss = total_loss / len(self.validation_dataloader)
            self.validation_losses.append(average_loss)
    
    # def test(self) -> torch.Tensor:
    #     self.model.eval()
    #     with torch.no_grad():
    #         outputs = self.model(self.testing_dataset[0])
    #         loss = self.criterion(outputs.reshape(-1), self.testing_dataset[1].reshape(-1))
    #         return loss

    def train(self, epochs: int) -> None:
        self.__prepare_testing_dataset__()
        for epoch in tqdm(range(epochs), desc='Epoch '):
            epochs_loss = []
            for inputs, targets in tqdm(self.training_dataloader, total = len(self.training_dataloader), desc='Training'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                epochs_loss.append(self.__train_step__(inputs, targets))
            self.trainin_losses.append(np.mean(epochs_loss))
            self.__validation__()
            self.__inform_progress__()
        
        # print(self.test())
        self.__get_plots__()

    def __get_plots__(self) -> None:
        plt.plot(self.trainin_losses, label='Training Loss')
        plt.plot(self.validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def __inform_progress__(self) -> None:
        print(f"Training Loss : {self.trainin_losses[-1]}, Validation Loss : {self.validation_losses[-1]}")

    def get_landscape(self) -> None:
            
            self.final_params = [param.clone().detach() for param in self.model.parameters()]
            self.initial_params = [param.clone().detach() for param in self.model.parameters()]

            alphas = np.linspace(-self.range_, self.range_, self.num_points)  
            betas = np.linspace(-self.range_, self.range_, self.num_points)
            alpha_beta_combinations = [(alpha, beta) for alpha in alphas for beta in betas]

            # Initialize losses array
            losses = np.zeros((len(alphas), len(betas)), dtype=np.float64)
            self.__prepare_landscape_set__()
            self.model.eval()

            for i, alpha_beta in enumerate(tqdm(alpha_beta_combinations, desc='Calculating Loss')):
                losses[i // len(betas), i % len(betas)] = self.__calculate_loss__(alpha_beta)[2]
                self.path_alphas.append(self.__calculate_loss__(alpha_beta)[0])
                self.path_betas.append(self.__calculate_loss__(alpha_beta)[1])
                self.path_losses.append(self.__calculate_loss__(alpha_beta)[2])
            
            self.model.eval()
            self.model = self.model.to("cpu")
            torch.save(self.model.state_dict(), "6_3_24_BU_BaseSequence.pth")

            plt.contour(alphas, betas, losses, cmap='rainbow')
            plt.xlabel('Alpha')
            plt.ylabel('Beta')
            plt.colorbar(label='Log Loss')
            plt.title('Contour Plot of Log Loss Landscape')

            # fig = plt.figure()
            # X_, Y_ = np.meshgrid(alphas, betas)
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_xlabel('Alpha')
            # ax.set_ylabel('Beta')
            # ax.set_zlabel('Log Loss')
            # ax.set_title('Contour Plot of Log Loss Landscape')

            # ax.plot_surface(X_, Y_, losses, cmap='rainbow')

            # self.path_alphas = np.array(self.path_alphas)
            # self.path_betas = np.array(self.path_betas)
            # self.path_losses = np.array(self.path_losses)

            plt.show()
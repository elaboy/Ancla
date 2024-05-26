from Ancla import HelperFunctions, Visualize, RegressiveModels, TransformationFunctions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Ancla import Featurizer
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.functional import F

batch_size = 256

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        # self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(32)
        # self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias = False)
        self.bn3 = nn.BatchNorm2d(64)
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias = False)
        self.bn4 = nn.BatchNorm2d(128)
        # self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias = False)
        self.bn5 = nn.BatchNorm2d(256)
        
        # Calculate the size of the flattened features after the last pooling layer
        self._to_linear = None
        self._get_to_linear_dim()

        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 1)
        # self.fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.2)

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
        x = self.fc2(x)
        # x = self.dropout(x)
        # x = self.fc3(x)
        
        return x
    
class RTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]).unsqueeze_(0), self.y[idx]
    
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
    model.load_state_dict(torch.load(r"D:\OtherPeptideResultsForTraining\RT_model_5_24_2024_V8.pth"))
    # Test model with predicting the fractions and making 10 plots with R^2

    # datasets
    fraction_datasets = []
    fractions_overlaps = []

    for fraction in [frac1, frac2, frac3, frac4, frac5, frac6, frac7, frac8, frac9, frac10]:
        
        X = fraction["BaseSequence"].tolist()
        y = fraction["Experimental"].tolist()

        #search in the training data for the X sequences and get the retention time
        overlaps = []
        import pandas as pd
        
        # Convert the training data to a dictionary for quick lookup
        sequence_to_time = pd.Series(training_data.ScanRetentionTime.values,
                                      index=training_data.BaseSequence).to_dict()

        # Find overlaps using the dictionary for fast lookup
        overlaps = [sequence_to_time[sequence] for sequence in X if sequence in sequence_to_time]

        fraction_features = Featurizer.featurize_all(X)

        # y = Featurizer.normalize_targets(y)

        fraction_dataset = RTDataset(fraction_features, y)

        fraction_datasets.append(fraction_dataset)
        fractions_overlaps.append(overlaps)
    
    model.eval()
    model.to("cuda")
    predictions = []
    for fraction in fraction_datasets:
        loader = DataLoader(fraction, batch_size=batch_size, shuffle=False)
        preds = []
        for X_batch, y_batch in loader:
            X_batch = X_batch.to("cuda").double()
            y_batch = y_batch.to("cuda").double()
            pred = model(X_batch)
            preds.append(pred.detach().cpu())
        predictions.append(torch.cat(preds))

    # plot the results and calculate R^2
    from sklearn.metrics import r2_score

    r2_scores = []

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    class PlotNavigator:
        def __init__(self, fraction_datasets, fractions_overlaps, predictions):
            self.fraction_datasets = fraction_datasets
            self.fractions_overlaps = fractions_overlaps
            self.predictions = predictions
            self.current_index = 0
            self.fig, self.axs = plt.subplots(1, 3, figsize=(12, 5))
            self.r2_scores = []
            
            self.next_button_ax = self.fig.add_axes([0.8, 0.01, 0.1, 0.05])
            self.next_button = Button(self.next_button_ax, 'Next')
            self.next_button.on_clicked(self.next_plot)
            
            self.prev_button_ax = self.fig.add_axes([0.7, 0.01, 0.1, 0.05])
            self.prev_button = Button(self.prev_button_ax, 'Previous')
            self.prev_button.on_clicked(self.prev_plot)
            
            self.plot_fraction(self.current_index)
            plt.show()

        def plot_fraction(self, index):
            self.axs[0].clear()
            self.axs[1].clear()
            
            fraction = self.fraction_datasets[index]
            actual_RT = self.fractions_overlaps[index]
            
            # scaler = StandardScaler()
            # scaler = scaler.fit(np.array(actual_RT).reshape(-1, 1))
            # actual_RT = scaler.transform(np.array(actual_RT).reshape(-1, 1))
            # predicted_RT = scaler.transform(self.predictions[index].detach().numpy().reshape(-1, 1))
            
            predicted_RT = self.predictions[index].detach().numpy()
            
            r2 = r2_score(actual_RT, predicted_RT)
            self.r2_scores.append(r2)
            
            self.axs[0].scatter(actual_RT, actual_RT, color="red", s=0.4, label="Actual")
            self.axs[0].scatter(fraction.y, actual_RT, color="blue", s=0.4, label="Predicted")
            self.axs[0].set_ylabel("Actual RT")
            self.axs[0].set_xlabel("Predicted RT")
            self.axs[0].set_title(f"Fraction {index + 1}")
            self.axs[0].text(0.1, 0.9, f"R^2: {r2:.3f}", ha="center", va="center", transform=self.axs[0].transAxes)

            from statsmodels.nonparametric.smoothers_lowess import lowess
            # make this a numpy vector 
            model = lowess(actual_RT, predicted_RT.squeeze(), frac=1./3, it = 4)
            
            #interpolate the lowess
            # f = scaler.transform(f.reshape(-1, 1))

            #plot the lowess
            r2 = r2_score(actual_RT, predicted_RT)
            self.axs[1].scatter(actual_RT, actual_RT, color='red', label='actual', s=0.4)
            self.axs[1].scatter(predicted_RT, actual_RT, color='blue', label='predictions', s=0.4)
            self.axs[1].text(0.1, 0.9, f"R^2: {r2:.3f}", ha="center", va="center", transform=self.axs[1].transAxes)
            self.axs[1].set_ylabel("Actual RT")
            self.axs[1].set_xlabel("Predicted RT")
            self.axs[1].set_title(f"Fraction {index + 1}")
            
            f = np.interp(predicted_RT.squeeze(), model[:, 0], model[:, 1])

            r2 = r2_score(actual_RT, f)
            self.axs[2].scatter(actual_RT, actual_RT, color='red', label='actual', s=0.4)
            self.axs[2].text(0.1, 0.9, f"R^2: {r2:.3f}", ha="center", va="center", transform=self.axs[2].transAxes)
            self.axs[2].scatter(f, actual_RT, color='green', s=0.4, label='calibrated predictions')
            self.axs[2].set_ylabel("Actual RT")
            self.axs[2].set_xlabel("Predicted RT")
            self.axs[2].set_title(f"Fraction {index + 1}")
            self.axs[2].legend()

            # self.axs[1].hist(actual_RT, bins=50, alpha=0.5, color="red", label="Actual RT")
            # self.axs[1].hist(predicted_RT, bins=50, alpha=0.5, color="blue", label="Predicted RT")
            # self.axs[1].legend()
            # self.axs[1].set_title(f"Fraction {index + 1}")
            
            self.fig.canvas.draw_idle()

        def next_plot(self, event):
            self.current_index = (self.current_index + 1) % len(self.fraction_datasets)
            self.plot_fraction(self.current_index)

        def prev_plot(self, event):
            self.current_index = (self.current_index - 1) % len(self.fraction_datasets)
            self.plot_fraction(self.current_index)

    # Assuming fraction_datasets, fractions_overlaps, and predictions are defined
    navigator = PlotNavigator(fraction_datasets, fractions_overlaps, predictions)

import torch
import torch.nn as nn
import pandas as pd

from analysis.preprocessing import split_cic_data
from sklearn.metrics import confusion_matrix, classification_report
from analysis.plotting import plot_confusion_matrix

from typing import Optional, Dict

class MLP(nn.Module):
    """
    Args:
        input_dim (int): Number of input features.
        hidden_dims (list of int): List with the number of units for each hidden layer.
        output_dim (int): Number of output features.
    """
    def __init__(self, data_path: str, width: int = 64):
        super(MLP, self).__init__()

        self.df = pd.read_csv(data_path)
        self._prepare_data()

        self.layer1 = nn.Linear(self.X_train.shape[1], width)
        self.layer2 = nn.Linear(width, width)
        self.layer3 = nn.Linear(width, width)
        self.layer4 = nn.Linear(width, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

    def loss(self, y_pred, y, class_weight):
        # Calculate squared errors
        squared_errors = (y - y_pred)**2

        if isinstance(y, bool) or isinstance(y_pred, bool):
            print(y)
            print(y_pred)
        
        # Determine weights for each example based on class
        weights = y * class_weight[1] + (1 - y) * class_weight[0]
        
        # Apply weights to squared errors
        weighted_squared_errors = weights * squared_errors
        
        # Return mean of weighted squared errors
        return torch.mean(weighted_squared_errors)
    
    def _prepare_data(self):
        splits = split_cic_data(self.df, 0.7, 0.15, 0.15)
        self.X_train = splits[0]
        self.X_test = splits[2]
        self.y_train = splits[3].ravel()
        self.y_test = splits[5].ravel()
    
    def train(self, class_weight: Dict[int, float], n_epochs: int = 1, batch_size: int = 32):

        # Define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(n_epochs):
            # Create mini-batches
            permutation = torch.randperm(self.X_train.shape[0])
            total_loss = 0
            
            for i in range(0, self.X_train.shape[0], batch_size):
                # Get batch indices
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = self.X_train[indices], self.y_train[indices]
                
                # Forward pass
                outputs = self(batch_x)
                
                # Compute loss
                loss = self.loss(outputs, batch_y.unsqueeze(1).float(), class_weight)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(indices)
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / self.X_train.size(0)
            
            # Print progress
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}')

    def evaluate(self, name: Optional[str] = None):
        """Evaluate the MLP model on test data"""
        
        # Set model to evaluation mode
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self(self.X_test)
            
            # For binary classification
            if outputs.shape[1] == 1:
                preds = (torch.sigmoid(outputs) >= 0.5).int().flatten()
            else:
                preds = torch.argmax(outputs, dim=1)
        
        # Convert to numpy for metrics
        y_true = self.y_test.cpu().numpy()
        y_pred = preds.cpu().numpy()
        
        print("MLP Results:")
        print(classification_report(y_true, y_pred))
        
        plot_confusion_matrix(
            confusion_matrix(y_true, y_pred),
            class_names=["Benign", "Malicious"],
            name=name
        )


if __name__ == "__main__":
    data_path = "data/Merged01.csv" 
    input_dim = 10
    hidden_dims = [32, 16]
    output_dim = 1

    model = MLP(data_path)
    model.train(class_weight={0: 15, 1:1})
    model.evaluate()
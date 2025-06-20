import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import math

class DeliveryTimeModel(nn.Module):
    """Neural network model for delivery time prediction"""
    
    def __init__(self, input_size: int):
        super(DeliveryTimeModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class DeliveryTimeDataset(Dataset):
    """Dataset for delivery time prediction"""
    
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class DeliveryTimePredictor:
    """
    Predicts delivery time based on various factors including:
    - Distance between driver and restaurant
    - Distance between restaurant and customer
    - Time of day
    - Day of week
    - Weather conditions
    - Traffic conditions
    - Restaurant preparation time
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def prepare_features(self, 
                        driver_location: Tuple[float, float],
                        restaurant_location: Tuple[float, float],
                        customer_location: Tuple[float, float],
                        time_of_day: float,
                        day_of_week: int,
                        weather_conditions: Dict,
                        traffic_conditions: Dict,
                        restaurant_prep_time: float) -> torch.Tensor:
        """
        Prepare features for prediction
        
        Args:
            driver_location: (x, y) coordinates of driver
            restaurant_location: (x, y) coordinates of restaurant
            customer_location: (x, y) coordinates of customer
            time_of_day: Hour of day (0-23)
            day_of_week: Day of week (0-6)
            weather_conditions: Dictionary containing weather features
            traffic_conditions: Dictionary containing traffic features
            restaurant_prep_time: Estimated preparation time in minutes
            
        Returns:
            Feature tensor for prediction
        """
        # Calculate distances
        driver_to_restaurant = torch.norm(
            torch.tensor(driver_location) - torch.tensor(restaurant_location)
        )
        restaurant_to_customer = torch.norm(
            torch.tensor(restaurant_location) - torch.tensor(customer_location)
        )
        
        # Prepare time features
        time_features = [
            time_of_day,
            torch.sin(torch.tensor(2 * math.pi * time_of_day / 24)),  # Cyclical encoding
            torch.cos(torch.tensor(2 * math.pi * time_of_day / 24)),
            day_of_week,
            torch.sin(torch.tensor(2 * math.pi * day_of_week / 7)),   # Cyclical encoding
            torch.cos(torch.tensor(2 * math.pi * day_of_week / 7))
        ]
        
        # Prepare weather features
        weather_features = [
            weather_conditions.get('temperature', 0),
            weather_conditions.get('precipitation', 0),
            weather_conditions.get('wind_speed', 0)
        ]
        
        # Prepare traffic features
        traffic_features = [
            traffic_conditions.get('congestion_level', 0),
            traffic_conditions.get('average_speed', 0)
        ]
        
        # Combine all features
        features = torch.tensor([
            driver_to_restaurant,
            restaurant_to_customer,
            *time_features,
            *weather_features,
            *traffic_features,
            restaurant_prep_time
        ], dtype=torch.float32)
        
        return features.unsqueeze(0)  # Add batch dimension
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, 
            batch_size: int = 32, epochs: int = 100,
            learning_rate: float = 0.001):
        """
        Train the model
        
        Args:
            X: Feature tensor
            y: Target delivery times
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        # Initialize model
        self.model = DeliveryTimeModel(X.shape[1]).to(self.device)
        
        # Create dataset and dataloader
        dataset = DeliveryTimeDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
    
    def predict(self, features: torch.Tensor) -> float:
        """
        Predict delivery time
        
        Args:
            features: Feature tensor
            
        Returns:
            Predicted delivery time in minutes
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            prediction = self.model(features)
            return prediction.item()
    
    def predict_with_confidence(self, features: torch.Tensor, 
                              num_samples: int = 100) -> Tuple[float, float]:
        """
        Predict delivery time with confidence interval using Monte Carlo dropout
        
        Args:
            features: Feature tensor
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (predicted time, confidence interval)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        self.model.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            features = features.to(self.device)
            for _ in range(num_samples):
                prediction = self.model(features)
                predictions.append(prediction.item())
        
        mean_prediction = np.mean(predictions)
        confidence = np.std(predictions)
        
        return mean_prediction, confidence 
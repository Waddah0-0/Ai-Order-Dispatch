import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os

class TalabatDataLoader:
    """
    Loads and processes Talabat order data for the simulation.
    """
    required_columns = [
        'Order_ID', 'Restaurant_ID', 'Restaurant_Lat', 'Restaurant_Lon',
        'Driver_ID', 'Driver_Lat', 'Driver_Lon', 'Driver_Availability',
        'Order_Time', 'Delivery_Time', 'Delivery_Duration_Minutes',
        'Delivery_Distance_km', 'Traffic_Level'
    ]
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        Args:
            data_path: Path to the Talabat orders CSV file
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data_path = data_path
        self.data = None
        self.restaurants = None
        self.drivers = None
    def load_data(self):
        """Load and preprocess the data"""
        try:
            self.data = pd.read_csv(self.data_path)
            missing_columns = [col for col in self.required_columns if col not in self.data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            self.data['Order_Time'] = pd.to_datetime(self.data['Order_Time'])
            self.data['Delivery_Time'] = pd.to_datetime(self.data['Delivery_Time'])
            self._validate_data()
            self.restaurants = self.data[['Restaurant_ID', 'Restaurant_Lat', 'Restaurant_Lon']].drop_duplicates()
            self.drivers = self.data[['Driver_ID', 'Driver_Lat', 'Driver_Lon', 'Driver_Availability']].drop_duplicates()
            print(f"Loaded {len(self.data)} orders")
            print(f"Found {len(self.restaurants)} unique restaurants")
            print(f"Found {len(self.drivers)} unique drivers")
            print("\nSample data:")
            print(self.data.head())
        except Exception as e:
            raise RuntimeError(f"Error loading data: {str(e)}")
    def _validate_data(self):
        """Validate data types and ranges"""
        missing_values = self.data[self.required_columns].isnull().sum()
        if missing_values.any():
            print("Warning: Found missing values:")
            print(missing_values[missing_values > 0])
        if not ((-90 <= self.data['Restaurant_Lat']) & (self.data['Restaurant_Lat'] <= 90)).all():
            raise ValueError("Invalid restaurant latitude values")
        if not ((-180 <= self.data['Restaurant_Lon']) & (self.data['Restaurant_Lon'] <= 180)).all():
            raise ValueError("Invalid restaurant longitude values")
        if not (self.data['Delivery_Duration_Minutes'] >= 0).all():
            raise ValueError("Negative delivery duration found")
        if not (self.data['Delivery_Distance_km'] >= 0).all():
            raise ValueError("Negative delivery distance found")
    def get_orders_for_time_window(self, 
                                 start_time: datetime,
                                 end_time: datetime) -> pd.DataFrame:
        """
        Get orders within a specific time window.
        Args:
            start_time: Start of the time window
            end_time: End of the time window
        Returns:
            DataFrame containing orders in the time window
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        if start_time >= end_time:
            raise ValueError("start_time must be before end_time")
        mask = (self.data['Order_Time'] >= start_time) & (self.data['Order_Time'] <= end_time)
        return self.data[mask].copy()
    def get_restaurant_locations(self) -> List[Tuple[float, float]]:
        """Get restaurant locations as (lat, lon) tuples"""
        if self.restaurants is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return list(zip(self.restaurants['Restaurant_Lat'], self.restaurants['Restaurant_Lon']))
    def get_driver_locations(self) -> List[Tuple[float, float]]:
        """Get driver locations as (lat, lon) tuples"""
        if self.drivers is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return list(zip(self.drivers['Driver_Lat'], self.drivers['Driver_Lon']))
    def get_order_details(self, order_id: int) -> Dict:
        """
        Get detailed information about a specific order.
        Args:
            order_id: ID of the order
        Returns:
            Dictionary containing order details
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        order = self.data[self.data['Order_ID'] == order_id]
        if len(order) == 0:
            raise ValueError(f"Order {order_id} not found")
        return order.iloc[0].to_dict()
    def get_available_drivers(self) -> pd.DataFrame:
        """Get currently available drivers"""
        if self.drivers is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.drivers[self.drivers['Driver_Availability'] == 'Online'].copy()
    def get_order_metrics(self) -> Dict:
        """Calculate key metrics from the order data"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return {
            'total_orders': len(self.data),
            'avg_delivery_time': self.data['Delivery_Duration_Minutes'].mean(),
            'avg_delivery_distance': self.data['Delivery_Distance_km'].mean(),
            'avg_order_value': self.data['Total_Price'].mean() if 'Total_Price' in self.data.columns else None,
            'traffic_levels': self.data['Traffic_Level'].value_counts().to_dict()
        }

if __name__ == "__main__":
    # Test the data loader
    try:
        from datetime import timedelta
        loader = TalabatDataLoader("data/talabat_enhanced_orders.csv")
        loader.load_data()
        
        # Print sample data
        print("\nSample Restaurant Locations:")
        print(loader.get_restaurant_locations()[:5])
        
        print("\nSample Driver Locations:")
        print(loader.get_driver_locations()[:5])
        
        # Get orders for the first hour
        start_time = loader.data['Order_Time'].min()
        end_time = start_time + timedelta(hours=1)
        orders = loader.get_orders_for_time_window(start_time, end_time)
        print(f"\nOrders in first hour: {len(orders)}")
        
        # Print metrics
        print("\nOrder Metrics:")
        metrics = loader.get_order_metrics()
        for key, value in metrics.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}") 
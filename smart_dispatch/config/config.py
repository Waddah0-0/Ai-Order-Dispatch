from typing import Dict, Any

class Config:
    """Configuration parameters for the smart dispatch system"""
    
    # System parameters
    GRID_SIZE: float = 20.0  # Size of the city grid
    MAX_DRIVERS: int = 100
    MAX_ORDERS: int = 50
    
    # Reinforcement Learning parameters
    RL_CONFIG: Dict[str, Any] = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 0.1,
        "batch_size": 64,
        "memory_size": 10000
    }
    
    # Predictive model parameters
    PREDICTIVE_CONFIG: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    }
    
    # Spatial optimization parameters
    SPATIAL_CONFIG: Dict[str, Any] = {
        "cluster_eps": 2.0,
        "min_cluster_samples": 3,
        "max_driver_distance": 5.0
    }
    
    # Simulation parameters
    SIMULATION_CONFIG: Dict[str, Any] = {
        "num_drivers": 50,
        "num_restaurants": 20,
        "simulation_duration": 24,  # hours
        "time_step": 1,  # minutes
        "order_generation_rate": 2  # orders per time step
    }
    
    # API parameters
    API_CONFIG: Dict[str, Any] = {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": True
    }
    
    # Weather and traffic parameters
    WEATHER_CONFIG: Dict[str, Any] = {
        "temperature_range": (-10, 40),  # Celsius
        "precipitation_range": (0, 100),  # mm
        "wind_speed_range": (0, 30)  # km/h
    }
    
    TRAFFIC_CONFIG: Dict[str, Any] = {
        "congestion_levels": ["low", "medium", "high"],
        "average_speed_range": (20, 60)  # km/h
    }
    
    # Restaurant parameters
    RESTAURANT_CONFIG: Dict[str, Any] = {
        "prep_time_range": (5, 30),  # minutes
        "capacity_range": (1, 10)  # concurrent orders
    }
    
    # Customer satisfaction parameters
    SATISFACTION_CONFIG: Dict[str, Any] = {
        "max_wait_time": 45,  # minutes
        "time_penalty_factor": 0.1,
        "distance_penalty_factor": 0.05
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get all configuration parameters"""
        return {
            "grid_size": cls.GRID_SIZE,
            "max_drivers": cls.MAX_DRIVERS,
            "max_orders": cls.MAX_ORDERS,
            "rl_config": cls.RL_CONFIG,
            "predictive_config": cls.PREDICTIVE_CONFIG,
            "spatial_config": cls.SPATIAL_CONFIG,
            "simulation_config": cls.SIMULATION_CONFIG,
            "api_config": cls.API_CONFIG,
            "weather_config": cls.WEATHER_CONFIG,
            "traffic_config": cls.TRAFFIC_CONFIG,
            "restaurant_config": cls.RESTAURANT_CONFIG,
            "satisfaction_config": cls.SATISFACTION_CONFIG
        }
    
    @classmethod
    def update_config(cls, new_config: Dict[str, Any]):
        """Update configuration parameters"""
        for key, value in new_config.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
            elif hasattr(cls, key):
                setattr(cls, key, value) 
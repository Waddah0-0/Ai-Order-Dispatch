import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional

class DispatchEnvironment(gym.Env):
    """
    Custom Environment for order-driver matching that follows gym interface.
    This environment simulates a food delivery dispatch system.
    """
    
    def __init__(self, config: Dict = None):
        super(DispatchEnvironment, self).__init__()
        
        # Configuration
        self.config = config or {}
        self.max_drivers = self.config.get('max_drivers', 100)
        self.max_orders = self.config.get('max_orders', 50)
        self.grid_size = self.config.get('grid_size', 20)  # City grid size
        
        # Validate configuration
        if self.max_drivers <= 0 or self.max_orders <= 0 or self.grid_size <= 0:
            raise ValueError("max_drivers, max_orders, and grid_size must be positive")
        
        # Define action and observation space
        # Action space: Assign order to driver (discrete)
        self.action_space = spaces.Discrete(self.max_drivers)
        
        # Observation space: State of orders and drivers
        # [driver_locations (100x2), driver_status (100), current_orders (100), 
        #  order_locations (50x2), order_status (50), prep_times (50), time (1)]
        obs_size = (self.max_drivers * 2 + self.max_drivers + self.max_drivers + 
                   self.max_orders * 2 + self.max_orders + self.max_orders + 1)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        try:
            # Set the seed if provided
            if seed is not None:
                np.random.seed(seed)
                torch.manual_seed(seed)
            
            # Initialize drivers
            self.drivers = {
                'locations': torch.rand(self.max_drivers, 2) * self.grid_size,
                'status': torch.zeros(self.max_drivers),  # 0: available, 1: busy
                'current_orders': torch.zeros(self.max_drivers)
            }
            
            # Initialize orders
            self.orders = {
                'locations': torch.rand(self.max_orders, 2) * self.grid_size,
                'status': torch.zeros(self.max_orders),  # 0: pending, 1: assigned, 2: completed
                'prep_times': torch.randint(5, 30, (self.max_orders,))
            }
            
            self.current_time = 0
            return self._get_observation(), {}
            
        except Exception as e:
            raise RuntimeError(f"Error resetting environment: {str(e)}")
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment
        
        Args:
            action: Index of driver to assign the current order to
            
        Returns:
            observation: Current state
            reward: Reward for the action
            done: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        try:
            # Get current pending order
            pending_orders = torch.where(self.orders['status'] == 0)[0]
            if len(pending_orders) == 0:
                return self._get_observation(), 0, True, False, {'message': 'No pending orders'}
            
            current_order = pending_orders[0]
            
            # Validate action
            if not isinstance(action, (int, np.integer)):
                raise ValueError(f"Action must be an integer, got {type(action)}")
            
            if action < 0 or action >= self.max_drivers:
                return self._get_observation(), -1, False, False, {'error': 'Invalid driver index'}
            
            if self.drivers['status'][action] == 1:
                return self._get_observation(), -1, False, False, {'error': 'Driver is busy'}
            
            # Assign order to driver
            self.drivers['status'][action] = 1
            self.drivers['current_orders'][action] = current_order
            self.orders['status'][current_order] = 1
            
            # Calculate reward
            reward = self._calculate_reward(action, current_order)
            
            # Update time and state
            self.current_time += 1
            self._update_state()
            
            # Check if episode is done
            done = len(pending_orders) == 1
            
            return self._get_observation(), reward, done, False, {}
            
        except Exception as e:
            raise RuntimeError(f"Error in step: {str(e)}")
    
    def _get_observation(self) -> np.ndarray:
        """Convert current state to observation vector"""
        try:
            obs = torch.cat([
                self.drivers['locations'].flatten(),
                self.drivers['status'],
                self.drivers['current_orders'],
                self.orders['locations'].flatten(),
                self.orders['status'],
                self.orders['prep_times'],
                torch.tensor([self.current_time], dtype=torch.float32)
            ])
            return obs.numpy()
        except Exception as e:
            raise RuntimeError(f"Error getting observation: {str(e)}")
    
    def _calculate_reward(self, driver_idx: int, order_idx: int) -> float:
        """Calculate reward for the action"""
        try:
            # Distance between driver and order
            distance = torch.norm(
                self.drivers['locations'][driver_idx] - self.orders['locations'][order_idx]
            )
            
            # Base reward is inverse of distance
            reward = 1.0 / (1.0 + distance.item())
            
            # Penalize if driver is already busy
            if self.drivers['status'][driver_idx] == 1:
                reward -= 1.0
            
            return reward
        except Exception as e:
            raise RuntimeError(f"Error calculating reward: {str(e)}")
    
    def _update_state(self):
        """Update the state of the environment"""
        try:
            # Update driver positions (simplified)
            self.drivers['locations'] += torch.randn_like(self.drivers['locations']) * 0.1
            self.drivers['locations'] = torch.clamp(self.drivers['locations'], 0, self.grid_size)
            
            # Update order status
            for driver_idx in range(self.max_drivers):
                if self.drivers['status'][driver_idx] == 1:
                    order_idx = int(self.drivers['current_orders'][driver_idx])
                    if self.orders['prep_times'][order_idx] <= 0:
                        self.orders['status'][order_idx] = 2  # Completed
                        self.drivers['status'][driver_idx] = 0  # Available
                    else:
                        self.orders['prep_times'][order_idx] -= 1 
        except Exception as e:
            raise RuntimeError(f"Error updating state: {str(e)}") 
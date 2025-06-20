import numpy as np
import torch
import logging
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from smart_dispatch.data.data_loader import TalabatDataLoader
from smart_dispatch.models.rl.environment import DispatchEnvironment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_start = datetime.now()
        
    def _on_step(self):
        if self.n_calls % 1000 == 0:
            elapsed = datetime.now() - self.training_start
            logging.info(f"Training step {self.n_calls}, elapsed time: {elapsed}")
        return True

def train_rl_agent(data_path: str, max_drivers: int = 100, max_orders: int = 50, 
                  grid_size: int = 20, total_timesteps: int = 100000,
                  save_dir: str = 'models'):
    """
    Train a PPO agent for order-driver matching using real data.
    
    Args:
        data_path: Path to the Talabat orders CSV file
        max_drivers: Maximum number of drivers in the environment
        max_orders: Maximum number of orders in the environment
        grid_size: Size of the city grid
        total_timesteps: Total number of timesteps for training
        save_dir: Directory to save the trained model
    """
    try:
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Load the Talabat data
        logging.info("Loading data...")
        data_loader = TalabatDataLoader(data_path)
        data_loader.load_data()
        
        # 2. Create the RL environment
        logging.info("Creating RL environment...")
        env = DispatchEnvironment(config={
            'max_drivers': max_drivers,
            'max_orders': max_orders,
            'grid_size': grid_size
        })
        env = DummyVecEnv([lambda: env])
        
        # 3. Initialize and train the PPO agent
        logging.info("Training PPO agent...")
        model = PPO('MlpPolicy', env, verbose=1)
        
        # Add callback for progress tracking
        callback = TrainingCallback()
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        # 4. Save the trained model
        model_path = os.path.join(save_dir, 'ppo_dispatch_model')
        logging.info(f"Saving model to {model_path}...")
        model.save(model_path)
        
        logging.info('Training complete!')
        
        # 5. Print training summary
        elapsed = datetime.now() - callback.training_start
        logging.info(f"Training Summary:")
        logging.info(f"- Total timesteps: {total_timesteps}")
        logging.info(f"- Total time: {elapsed}")
        logging.info(f"- Model saved to: {model_path}")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_rl_agent(
            data_path='data/talabat_enhanced_orders.csv',
            max_drivers=100,
            max_orders=50,
            grid_size=20,
            total_timesteps=100000,
            save_dir='models'
        )
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise 
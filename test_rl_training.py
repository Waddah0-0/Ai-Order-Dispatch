import numpy as np
import torch
import logging
import os
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from smart_dispatch.data.data_loader import TalabatDataLoader
from smart_dispatch.models.rl.environment import DispatchEnvironment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_training.log'),
        logging.StreamHandler()
    ]
)

def test_rl_training():
    """Test the RL training pipeline with a small dataset"""
    try:
        # 1. Load a small subset of data
        logging.info("Loading test data...")
        data_loader = TalabatDataLoader('data/talabat_enhanced_orders.csv')
        data_loader.load_data()
        
        # Get first hour of data
        start_time = data_loader.data['Order_Time'].min()
        end_time = start_time + timedelta(hours=1)
        test_data = data_loader.get_orders_for_time_window(start_time, end_time)
        
        logging.info(f"Using {len(test_data)} orders for testing")
        
        # 2. Create a small environment
        logging.info("Creating test environment...")
        env = DispatchEnvironment(config={
            'max_drivers': 10,  # Small number for testing
            'max_orders': 5,    # Small number for testing
            'grid_size': 20
        })
        env = DummyVecEnv([lambda: env])
        
        # 3. Train for a short duration
        logging.info("Training PPO agent (test run)...")
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=1000)  # Short training for testing
        
        # 4. Test the model
        logging.info("Testing model predictions...")
        obs = env.reset()
        total_reward = 0.0
        done = False
        step = 0
        
        while not done and step < 50:  # Limit test steps
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])  # Convert numpy array to float
            step += 1
            
            if 'error' in info:
                logging.warning(f"Step {step}: {info['error']}")
            else:
                logging.info(f"Step {step}: Reward = {float(reward[0]):.2f}")
        
        logging.info(f"Test completed:")
        logging.info(f"- Total steps: {step}")
        logging.info(f"- Total reward: {total_reward:.2f}")
        logging.info(f"- Average reward: {total_reward/step:.2f}")
        
        # 5. Save test results
        results = {
            'total_steps': step,
            'total_reward': total_reward,
            'avg_reward': total_reward/step,
            'test_data_size': len(test_data)
        }
        
        # Save results to file
        with open('test_results.txt', 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        logging.info("Test results saved to test_results.txt")
        
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_rl_training() 
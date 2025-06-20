import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random
from ..models.rl.environment import DispatchEnvironment
from ..models.predictive.delivery_time_predictor import DeliveryTimePredictor
from ..models.graph.spatial_optimizer import SpatialOptimizer

class DeliverySimulator:
    """
    Simulates a food delivery system with multiple drivers, restaurants, and customers.
    """
    
    def __init__(self, 
                 num_drivers: int = 50,
                 num_restaurants: int = 20,
                 grid_size: float = 20.0,
                 time_step: int = 60,  
                 order_generation_rate: float = 0.1):  
        self.num_drivers = num_drivers
        self.num_restaurants = num_restaurants
        self.grid_size = grid_size
        self.time_step = time_step
        self.order_generation_rate = order_generation_rate
        self.dispatch_env = DispatchEnvironment()
        self.delivery_predictor = DeliveryTimePredictor()
        self.spatial_optimizer = SpatialOptimizer(grid_size=grid_size)
        self.current_time = datetime.now()
        self.orders: List[Dict] = []
        self.metrics_history: List[Dict] = []
        self.active_drivers = set()
    
    def initialize_simulation(self):
        """Initialize the simulation with random locations"""
        for _ in range(self.num_restaurants):
            coords = (
                random.uniform(0, self.grid_size),
                random.uniform(0, self.grid_size)
            )
            self.spatial_optimizer.add_location(
                coordinates=coords,
                location_type="restaurant",
                status="available"
            )
        for _ in range(self.num_drivers):
            coords = (
                random.uniform(0, self.grid_size),
                random.uniform(0, self.grid_size)
            )
            self.spatial_optimizer.add_location(
                coordinates=coords,
                location_type="driver",
                status="available"
            )
    def generate_order(self) -> Dict:
        """Generate a random order"""
        restaurant_id = random.choice([
            loc_id for loc_id, loc in self.spatial_optimizer.locations.items()
            if loc.type == "restaurant"
        ])
        customer_coords = (
            random.uniform(0, self.grid_size),
            random.uniform(0, self.grid_size)
        )
        prep_time = random.uniform(5, 30)
        return {
            "restaurant_id": restaurant_id,
            "customer_location": customer_coords,
            "prep_time": prep_time,
            "created_at": self.current_time,
            "status": "pending"
        }
    def simulate_time_step(self):
        """Simulate one time step"""
        self.current_time += timedelta(seconds=self.time_step)
        num_new_orders = np.random.poisson(self.order_generation_rate * (self.time_step / 60))
        for _ in range(num_new_orders):
            order = self.generate_order()
            self.orders.append(order)
            try:
                customer_id = self.spatial_optimizer.add_location(
                    coordinates=order["customer_location"],
                    location_type="customer",
                    status="pending"
                )
                nearest_driver = self.spatial_optimizer.find_nearest_driver(
                    order["restaurant_id"],
                    max_distance=10.0
                )
                if nearest_driver is not None and nearest_driver not in self.active_drivers:
                    self.spatial_optimizer.update_location(
                        location_id=nearest_driver,
                        new_status="busy"
                    )
                    self.active_drivers.add(nearest_driver)
                    cluster = [order["restaurant_id"], customer_id]
                    route = self.spatial_optimizer.optimize_routes(cluster)
                    order["status"] = "assigned"
                    order["driver_id"] = nearest_driver
                    order["route"] = route
                    order["assigned_at"] = self.current_time
            except Exception as e:
                print(f"Failed to dispatch order: {e}")
        completed_orders = []
        for order in self.orders:
            if order["status"] == "assigned":
                driver_id = order["driver_id"]
                driver = self.spatial_optimizer.locations[driver_id]
                new_coords = (
                    driver.coordinates[0] + random.uniform(-0.1, 0.1),
                    driver.coordinates[1] + random.uniform(-0.1, 0.1)
                )
                self.spatial_optimizer.update_location(
                    location_id=driver_id,
                    new_coordinates=new_coords
                )
                time_since_assignment = (self.current_time - order["assigned_at"]).total_seconds() / 60
                if time_since_assignment >= order["prep_time"]:
                    order["status"] = "completed"
                    order["completed_at"] = self.current_time
                    completed_orders.append(order)
                    self.spatial_optimizer.update_location(
                        location_id=driver_id,
                        new_status="available"
                    )
                    self.active_drivers.remove(driver_id)
        self.orders = [order for order in self.orders if order["status"] != "completed"]
        metrics = self.spatial_optimizer.calculate_metrics()
        metrics["timestamp"] = self.current_time
        metrics["active_orders"] = len(self.orders)
        metrics["completed_orders"] = len(completed_orders)
        self.metrics_history.append(metrics)
    def run_simulation(self, duration: int = 3600):
        """Run the complete simulation"""
        self.initialize_simulation()
        end_time = self.current_time + timedelta(seconds=duration)
        while self.current_time < end_time:
            self.simulate_time_step()
            if self.current_time.minute == 0 and self.current_time.second == 0:
                print(f"Simulation time: {self.current_time}")
                print(f"Active orders: {len(self.orders)}")
                print(f"Available drivers: {self.metrics_history[-1]['available_drivers']}")
                print("---")
    def get_simulation_results(self) -> Dict:
        """Get simulation results and metrics"""
        completed_orders = [order for order in self.orders if order.get("status") == "completed"]
        total_delivery_times = [
            (order["completed_at"] - order["created_at"]).total_seconds() / 60
            for order in completed_orders
        ]
        return {
            "total_orders": len(self.orders) + len(completed_orders),
            "completed_orders": len(completed_orders),
            "average_delivery_time": np.mean(total_delivery_times) if total_delivery_times else 0,
            "driver_utilization": len(self.active_drivers) / self.num_drivers,
            "metrics_history": self.metrics_history
        }

if __name__ == "__main__":
    # Run a sample simulation
    simulator = DeliverySimulator(
        num_drivers=50,
        num_restaurants=20,
        grid_size=20.0,
        time_step=60,  # 1-minute time steps
        order_generation_rate=0.1  # 1 order per 10 minutes on average
    )
    simulator.run_simulation(duration=3600)  # 1 hour
    results = simulator.get_simulation_results()
    print("\nSimulation Results:")
    print(f"Total Orders: {results['total_orders']}")
    print(f"Completed Orders: {results['completed_orders']}")
    print(f"Average Delivery Time: {results['average_delivery_time']:.2f} minutes")
    print(f"Driver Utilization: {results['driver_utilization']:.2%}") 
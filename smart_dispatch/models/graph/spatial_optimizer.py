import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Location:
    id: int
    coordinates: Tuple[float, float]
    type: str  
    status: str  

class SpatialOptimizer:
    """
    Optimizes spatial distribution of orders and drivers using graph theory
    and clustering algorithms.
    """
    
    def __init__(self, grid_size: float = 20.0):
        self.grid_size = grid_size
        self.graph = nx.Graph()
        self.locations: Dict[int, Location] = {}
        self.next_id = 0
    
    def add_location(self, coordinates: Tuple[float, float], 
                    location_type: str, status: str = 'available') -> int:
        """
        Add a new location to the spatial graph
        
        Args:
            coordinates: (x, y) coordinates
            location_type: Type of location ('restaurant', 'customer', or 'driver')
            status: Current status of the location
            
        Returns:
            ID of the added location
        """
        location_id = self.next_id
        self.next_id += 1
        
        location = Location(
            id=location_id,
            coordinates=coordinates,
            type=location_type,
            status=status
        )
        
        self.locations[location_id] = location
        self.graph.add_node(location_id, **location.__dict__)
        
        return location_id
    
    def update_location(self, location_id: int, 
                       new_coordinates: Optional[Tuple[float, float]] = None,
                       new_status: Optional[str] = None):
        """
        Update location information
        
        Args:
            location_id: ID of the location to update
            new_coordinates: New coordinates (optional)
            new_status: New status (optional)
        """
        if location_id not in self.locations:
            raise ValueError(f"Location {location_id} not found")
        
        location = self.locations[location_id]
        
        if new_coordinates is not None:
            location.coordinates = new_coordinates
        if new_status is not None:
            location.status = new_status
        
        
        self.graph.nodes[location_id].update(location.__dict__)
    
    def find_optimal_clusters(self, eps: float = 2.0, min_samples: int = 3) -> List[List[int]]:
        """
        Find optimal clusters of locations using DBSCAN
        
        Args:
            eps: Maximum distance between points in a cluster
            min_samples: Minimum number of points required to form a cluster
            
        Returns:
            List of clusters, where each cluster is a list of location IDs
        """
        
        coordinates = np.array([loc.coordinates for loc in self.locations.values()])
        
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        
        
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(list(self.locations.keys())[idx])
        
        return list(clusters.values())
    
    def find_nearest_driver(self, location_id: int, 
                          max_distance: float = 5.0) -> Optional[int]:
        """
        Find the nearest available driver to a location
        
        Args:
            location_id: ID of the location to find driver for
            max_distance: Maximum allowed distance
            
        Returns:
            ID of the nearest available driver, or None if none found
        """
        if location_id not in self.locations:
            raise ValueError(f"Location {location_id} not found")
        
        target_location = self.locations[location_id]
        nearest_driver = None
        min_distance = float('inf')
        
        for loc_id, location in self.locations.items():
            if (location.type == 'driver' and 
                location.status == 'available'):
                distance = np.linalg.norm(
                    np.array(target_location.coordinates) - 
                    np.array(location.coordinates)
                )
                if distance < min_distance and distance <= max_distance:
                    min_distance = distance
                    nearest_driver = loc_id
        
        return nearest_driver
    
    def optimize_routes(self, cluster: List[int]) -> List[Tuple[int, int]]:
        """
        Optimize delivery routes within a cluster using TSP
        
        Args:
            cluster: List of location IDs in the cluster
            
        Returns:
            List of (from_id, to_id) pairs representing the optimal route
        """
        if not cluster:
            return []
        
        
        subgraph = self.graph.subgraph(cluster)
        
        
        n = len(cluster)
        distance_matrix = np.zeros((n, n))
        for i, loc1_id in enumerate(cluster):
            for j, loc2_id in enumerate(cluster):
                if i != j:
                    coords1 = self.locations[loc1_id].coordinates
                    coords2 = self.locations[loc2_id].coordinates
                    distance_matrix[i, j] = np.linalg.norm(
                        np.array(coords1) - np.array(coords2)
                    )
        
        
        tsp = nx.approximation.traveling_salesman_problem
        route = tsp(subgraph, weight='distance')
        
        
        edges = []
        for i in range(len(route) - 1):
            edges.append((route[i], route[i + 1]))
        
        return edges
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate various metrics about the current state
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'total_locations': len(self.locations),
            'available_drivers': sum(1 for loc in self.locations.values() 
                                  if loc.type == 'driver' and loc.status == 'available'),
            'pending_orders': sum(1 for loc in self.locations.values() 
                                if loc.type == 'customer' and loc.status == 'pending'),
            'average_distance': 0.0
        }
        
        
        if self.graph.edges():
            distances = []
            for u, v in self.graph.edges():
                coords1 = self.locations[u].coordinates
                coords2 = self.locations[v].coordinates
                distances.append(np.linalg.norm(
                    np.array(coords1) - np.array(coords2)
                ))
            metrics['average_distance'] = np.mean(distances)
        
        return metrics 
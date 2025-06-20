from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Tuple
from ..models.rl.environment import DispatchEnvironment
from ..models.predictive.delivery_time_predictor import DeliveryTimePredictor
from ..models.graph.spatial_optimizer import SpatialOptimizer

app = FastAPI(title="Smart Dispatch AI System")


dispatch_env = DispatchEnvironment()
delivery_predictor = DeliveryTimePredictor()
spatial_optimizer = SpatialOptimizer()

class LocationRequest(BaseModel):
    coordinates: Tuple[float, float]
    location_type: str
    status: str = "available"

class DeliveryPredictionRequest(BaseModel):
    driver_location: Tuple[float, float]
    restaurant_location: Tuple[float, float]
    customer_location: Tuple[float, float]
    time_of_day: float
    day_of_week: int
    weather_conditions: Dict
    traffic_conditions: Dict
    restaurant_prep_time: float

class DispatchRequest(BaseModel):
    order_id: int
    restaurant_location: Tuple[float, float]
    customer_location: Tuple[float, float]
    restaurant_prep_time: float

@app.post("/locations")
async def add_location(location: LocationRequest):
    """Add a new location to the spatial optimizer"""
    try:
        location_id = spatial_optimizer.add_location(
            coordinates=location.coordinates,
            location_type=location.location_type,
            status=location.status
        )
        return {"location_id": location_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-delivery-time")
async def predict_delivery_time(request: DeliveryPredictionRequest):
    """Predict delivery time for a given order"""
    try:
        features = delivery_predictor.prepare_features(
            driver_location=request.driver_location,
            restaurant_location=request.restaurant_location,
            customer_location=request.customer_location,
            time_of_day=request.time_of_day,
            day_of_week=request.day_of_week,
            weather_conditions=request.weather_conditions,
            traffic_conditions=request.traffic_conditions,
            restaurant_prep_time=request.restaurant_prep_time
        )
        
        predicted_time, confidence = delivery_predictor.predict_with_confidence(features)
        
        return {
            "predicted_time": predicted_time,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/dispatch")
async def dispatch_order(request: DispatchRequest):
    """Dispatch an order to the optimal driver"""
    try:
        
        restaurant_id = spatial_optimizer.add_location(
            coordinates=request.restaurant_location,
            location_type="restaurant",
            status="pending"
        )
        
        customer_id = spatial_optimizer.add_location(
            coordinates=request.customer_location,
            location_type="customer",
            status="pending"
        )
        
        
        nearest_driver = spatial_optimizer.find_nearest_driver(restaurant_id)
        
        if nearest_driver is None:
            raise HTTPException(
                status_code=404,
                detail="No available drivers found within range"
            )
        
        
        spatial_optimizer.update_location(
            location_id=nearest_driver,
            new_status="busy"
        )
        
        
        cluster = [restaurant_id, customer_id]
        route = spatial_optimizer.optimize_routes(cluster)
        
        return {
            "driver_id": nearest_driver,
            "route": route,
            "restaurant_id": restaurant_id,
            "customer_id": customer_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get current system metrics"""
    try:
        return spatial_optimizer.calculate_metrics()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/clusters")
async def get_clusters(eps: float = 2.0, min_samples: int = 3):
    """Get current location clusters"""
    try:
        clusters = spatial_optimizer.find_optimal_clusters(
            eps=eps,
            min_samples=min_samples
        )
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
"""
REST API for NASA Ground Operations Optimizer.

Provides endpoints for:
- Running optimizations
- Retrieving schedule results
- What-if scenario analysis
- Performance monitoring

Usage:
    uvicorn api.app:app --reload
    
Author: Operations Research Portfolio Project
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

app = FastAPI(
    title="NASA Ground Operations Optimizer API",
    description="API for optimizing ground operations scheduling",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"


class OptimizationRequest(BaseModel):
    objective: str = "minimize_makespan"
    formulation: str = "disjunctive"
    time_limit: int = 3600
    use_heuristic_warm_start: bool = True


class ScenarioRequest(BaseModel):
    params: Dict[str, Any]
    name: str = "custom"


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "NASA Ground Operations Optimizer API", "version": "1.0.0"}


@app.get("/missions")
async def get_missions():
    """Get all missions."""
    with open(DATA_DIR / "missions.json", 'r') as f:
        data = json.load(f)
    return data


@app.get("/resources")
async def get_resources():
    """Get all resources."""
    with open(DATA_DIR / "resources.json", 'r') as f:
        data = json.load(f)
    return data


@app.get("/constraints")
async def get_constraints():
    """Get constraint configuration."""
    with open(DATA_DIR / "constraints.json", 'r') as f:
        data = json.load(f)
    return data


@app.post("/optimize")
async def optimize(request: OptimizationRequest):
    """Run optimization and return results."""
    try:
        if request.formulation == "time_indexed":
            from src.model.time_indexed import TimeIndexedScheduler
            scheduler = TimeIndexedScheduler(data_dir=str(DATA_DIR))
        else:
            from src.model.scheduler import GroundOpsScheduler
            scheduler = GroundOpsScheduler(data_dir=str(DATA_DIR))
            
        scheduler.load_data()
        
        # Use heuristic warm start if requested
        if request.use_heuristic_warm_start:
            from src.model.heuristics import generate_warm_start
            warm_start = generate_warm_start(
                scheduler.tasks, scheduler.resources,
                scheduler.planning_start, scheduler.time_horizon_hours
            )
            # Could pass warm start to solver here
            
        result = scheduler.optimize(objective=request.objective)
        
        return {
            "status": result.status,
            "makespan_hours": result.makespan_hours,
            "total_cost": result.total_cost,
            "solve_time_seconds": result.solve_time_seconds,
            "mip_gap": result.mip_gap,
            "tasks": result.tasks.to_dict(orient='records') if len(result.tasks) > 0 else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/heuristic")
async def run_heuristic(method: str = "multi_start"):
    """Run heuristic scheduling."""
    try:
        from src.model.heuristics import HeuristicScheduler
        scheduler = HeuristicScheduler(data_dir=str(DATA_DIR))
        scheduler.load_data()
        result = scheduler.schedule(method=method)
        
        return {
            "heuristic": result.heuristic_name,
            "makespan_hours": result.makespan_hours,
            "total_cost": result.total_cost,
            "solve_time_seconds": result.solve_time_seconds,
            "feasible": result.is_feasible,
            "tasks": result.schedule.to_dict(orient='records') if len(result.schedule) > 0 else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario")
async def run_scenario(request: ScenarioRequest):
    """Run what-if scenario analysis."""
    try:
        from src.simulation.what_if import WhatIfAnalyzer
        analyzer = WhatIfAnalyzer(base_data_dir=str(DATA_DIR))
        analyzer.load_base_case()
        result = analyzer.run_scenario(request.params, request.name)
        
        return {
            "scenario_name": result.scenario_name,
            "parameter_changes": result.parameter_changes,
            "base_makespan": result.base_result.makespan_hours,
            "scenario_makespan": result.scenario_result.makespan_hours,
            "metrics_delta": result.metrics_delta
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare-heuristics")
async def compare_heuristics():
    """Compare all heuristic methods."""
    try:
        from src.model.heuristics import HeuristicScheduler
        scheduler = HeuristicScheduler(data_dir=str(DATA_DIR))
        scheduler.load_data()
        df = scheduler.compare_methods()
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/summary")
async def performance_summary():
    """Get performance history summary."""
    from src.model.performance import PerformanceTracker
    tracker = PerformanceTracker()
    return tracker.get_summary_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
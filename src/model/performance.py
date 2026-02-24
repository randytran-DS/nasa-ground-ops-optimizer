"""
Performance monitoring and profiling for NASA Ground Operations Scheduler.

This module provides tools for tracking optimization performance,
identifying bottlenecks, and collecting statistics for analysis.

Features:
- Solve time tracking and breakdown
- Model size statistics (variables, constraints)
- Memory usage monitoring
- Performance profiling and bottleneck identification
- Historical performance tracking

Author: Operations Research Portfolio Project
"""

import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import functools


@dataclass
class ModelStatistics:
    """Statistics about the optimization model size."""
    num_variables: int = 0
    num_binary_variables: int = 0
    num_integer_variables: int = 0
    num_continuous_variables: int = 0
    num_constraints: int = 0
    num_linear_constraints: int = 0
    num_quadratic_constraints: int = 0
    num_sos_constraints: int = 0
    num_nonzeros: int = 0  # Non-zeros in constraint matrix
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SolveStatistics:
    """Statistics from the optimization solve."""
    solve_time_seconds: float = 0.0
    presolve_time_seconds: float = 0.0
    simplex_iterations: int = 0
    barrier_iterations: int = 0
    node_count: int = 0
    obj_value: float = 0.0
    mip_gap: float = 0.0
    best_bound: float = 0.0
    status: str = "unknown"
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class MemoryStatistics:
    """Memory usage statistics during optimization."""
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PerformanceReport:
    """Comprehensive performance report for an optimization run."""
    timestamp: str
    objective_type: str
    model_stats: ModelStatistics
    solve_stats: SolveStatistics
    memory_stats: MemoryStatistics
    num_tasks: int = 0
    num_resources: int = 0
    planning_horizon_hours: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "objective_type": self.objective_type,
            "num_tasks": self.num_tasks,
            "num_resources": self.num_resources,
            "planning_horizon_hours": self.planning_horizon_hours,
            "model_statistics": self.model_stats.to_dict(),
            "solve_statistics": self.solve_stats.to_dict(),
            "memory_statistics": self.memory_stats.to_dict(),
            "custom_metrics": self.custom_metrics
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "PERFORMANCE REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Objective: {self.objective_type}",
            "",
            "Problem Size:",
            f"  Tasks: {self.num_tasks}",
            f"  Resources: {self.num_resources}",
            f"  Planning Horizon: {self.planning_horizon_hours} hours",
            "",
            "Model Statistics:",
            f"  Variables: {self.model_stats.num_variables:,}",
            f"    - Binary: {self.model_stats.num_binary_variables:,}",
            f"    - Integer: {self.model_stats.num_integer_variables:,}",
            f"    - Continuous: {self.model_stats.num_continuous_variables:,}",
            f"  Constraints: {self.model_stats.num_constraints:,}",
            f"  Non-zeros: {self.model_stats.num_nonzeros:,}",
            "",
            "Solve Statistics:",
            f"  Status: {self.solve_stats.status}",
            f"  Solve Time: {self.solve_stats.solve_time_seconds:.2f} seconds",
            f"  Objective Value: {self.solve_stats.obj_value:.2f}",
            f"  MIP Gap: {self.solve_stats.mip_gap:.4%}",
            f"  Nodes Explored: {self.solve_stats.node_count:,}",
            "",
            "Memory Usage:",
            f"  Peak: {self.memory_stats.peak_memory_mb:.1f} MB",
            f"  Average: {self.memory_stats.avg_memory_mb:.1f} MB",
        ]
        return "\n".join(lines)


class PerformanceMonitor:
    """
    Monitor for tracking optimization performance.
    
    Usage:
        monitor = PerformanceMonitor()
        monitor.start()
        # ... run optimization ...
        monitor.stop()
        report = monitor.get_report(model, scheduler)
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.memory_samples: List[float] = []
        self.process = psutil.Process()
        self._sampling = False
        
    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.memory_samples = []
        self.memory_before_mb = self.process.memory_info().rss / 1024 / 1024
        self._sampling = True
        
    def sample_memory(self):
        """Take a memory sample."""
        if self._sampling:
            mem_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(mem_mb)
            
    def stop(self):
        """Stop monitoring."""
        self.end_time = time.time()
        self.memory_after_mb = self.process.memory_info().rss / 1024 / 1024
        self._sampling = False
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
        
    def get_memory_stats(self) -> MemoryStatistics:
        """Get memory statistics."""
        if not self.memory_samples:
            return MemoryStatistics()
            
        return MemoryStatistics(
            peak_memory_mb=max(self.memory_samples),
            avg_memory_mb=sum(self.memory_samples) / len(self.memory_samples),
            memory_before_mb=getattr(self, 'memory_before_mb', 0),
            memory_after_mb=getattr(self, 'memory_after_mb', 0)
        )
        
    def get_report(
        self,
        model,
        scheduler,
        objective_type: str = "unknown"
    ) -> PerformanceReport:
        """
        Generate a comprehensive performance report.
        
        Args:
            model: The Gurobi model
            scheduler: The GroundOpsScheduler instance
            objective_type: The optimization objective used
            
        Returns:
            PerformanceReport with all statistics
        """
        # Extract model statistics
        model_stats = self._extract_model_stats(model)
        
        # Extract solve statistics
        solve_stats = self._extract_solve_stats(model)
        
        # Get memory stats
        memory_stats = self.get_memory_stats()
        
        return PerformanceReport(
            timestamp=datetime.now().isoformat(),
            objective_type=objective_type,
            model_stats=model_stats,
            solve_stats=solve_stats,
            memory_stats=memory_stats,
            num_tasks=len(scheduler.tasks),
            num_resources=len(scheduler.resources),
            planning_horizon_hours=scheduler.time_horizon_hours
        )
        
    def _extract_model_stats(self, model) -> ModelStatistics:
        """Extract statistics from the Gurobi model."""
        if model is None:
            return ModelStatistics()
            
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            stats = ModelStatistics()
            stats.num_variables = model.NumVars
            stats.num_constraints = model.NumConstrs
            
            # Count variable types
            for var in model.getVars():
                if var.VType == GRB.BINARY:
                    stats.num_binary_variables += 1
                elif var.VType == GRB.INTEGER:
                    stats.num_integer_variables += 1
                else:
                    stats.num_continuous_variables += 1
                    
            # Get non-zero count
            stats.num_nonzeros = model.NumNZs
            
            return stats
            
        except Exception as e:
            print(f"Warning: Could not extract model stats: {e}")
            return ModelStatistics()
            
    def _extract_solve_stats(self, model) -> SolveStatistics:
        """Extract solve statistics from the Gurobi model."""
        if model is None:
            return SolveStatistics()
            
        try:
            import gurobipy as gp
            from gurobipy import GRB
            
            stats = SolveStatistics()
            stats.solve_time_seconds = model.Runtime
            stats.obj_value = model.ObjVal if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT else 0
            stats.mip_gap = model.MIPGap if hasattr(model, 'MIPGap') else 0
            stats.best_bound = model.ObjBound if hasattr(model, 'ObjBound') else 0
            stats.node_count = model.NodeCount if hasattr(model, 'NodeCount') else 0
            stats.simplex_iterations = getattr(model, 'IterCount', 0)
            
            # Status mapping
            status_map = {
                GRB.OPTIMAL: "optimal",
                GRB.INFEASIBLE: "infeasible",
                GRB.TIME_LIMIT: "time_limit",
                GRB.UNBOUNDED: "unbounded",
                GRB.INF_OR_UNBD: "infeasible_or_unbounded"
            }
            stats.status = status_map.get(model.Status, f"unknown_{model.Status}")
            
            return stats
            
        except Exception as e:
            print(f"Warning: Could not extract solve stats: {e}")
            return SolveStatistics()


class PerformanceTracker:
    """
    Track performance across multiple optimization runs.
    
    Stores historical performance data for analysis and comparison.
    """
    
    def __init__(self, storage_path: str = "performance_history.json"):
        """
        Initialize the performance tracker.
        
        Args:
            storage_path: Path to JSON file for storing history
        """
        self.storage_path = Path(storage_path)
        self.history: List[PerformanceReport] = []
        self._load_history()
        
    def _load_history(self):
        """Load performance history from file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    # Note: We store as dicts for JSON serialization
                    self.history = data
            except Exception as e:
                print(f"Warning: Could not load performance history: {e}")
                self.history = []
                
    def record(self, report: PerformanceReport):
        """Record a performance report."""
        self.history.append(report.to_dict())
        self._save_history()
        
    def _save_history(self):
        """Save performance history to file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save performance history: {e}")
            
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all runs."""
        if not self.history:
            return {}
            
        solve_times = [h.get('solve_statistics', {}).get('solve_time_seconds', 0) 
                       for h in self.history]
        model_sizes = [h.get('model_statistics', {}).get('num_variables', 0) 
                       for h in self.history]
        
        return {
            "total_runs": len(self.history),
            "avg_solve_time": sum(solve_times) / len(solve_times),
            "max_solve_time": max(solve_times) if solve_times else 0,
            "min_solve_time": min(solve_times) if solve_times else 0,
            "avg_model_size": sum(model_sizes) / len(model_sizes) if model_sizes else 0,
        }
        
    def clear_history(self):
        """Clear the performance history."""
        self.history = []
        self._save_history()


def profile_optimization(func):
    """
    Decorator to profile optimization functions.
    
    Usage:
        @profile_optimization
        def run_optimization():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = PerformanceMonitor()
        monitor.start()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            monitor.stop()
            elapsed = monitor.get_elapsed_time()
            mem_stats = monitor.get_memory_stats()
            print(f"\n[Profile] {func.__name__}")
            print(f"  Time: {elapsed:.2f} seconds")
            print(f"  Peak Memory: {mem_stats.peak_memory_mb:.1f} MB")
            
    return wrapper


def benchmark_model_sizes(
    scheduler_class,
    task_counts: List[int] = [10, 20, 50, 100],
    output_path: str = "benchmark_results.json"
) -> Dict:
    """
    Benchmark model building and solving for different problem sizes.
    
    Args:
        scheduler_class: The scheduler class to benchmark
        task_counts: List of task counts to test
        output_path: Path to save benchmark results
        
    Returns:
        Dictionary with benchmark results
    """
    results = []
    
    for n_tasks in task_counts:
        print(f"\nBenchmarking with {n_tasks} tasks...")
        
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Note: This would need a data generator to create test instances
        # For now, just track time
        try:
            # Placeholder for actual benchmark
            elapsed = monitor.get_elapsed_time()
            results.append({
                "num_tasks": n_tasks,
                "status": "skipped",
                "message": "Data generator needed for benchmarking"
            })
        except Exception as e:
            results.append({
                "num_tasks": n_tasks,
                "status": "error",
                "error": str(e)
            })
            
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    return {"results": results}


# Utility functions
def print_model_info(model):
    """Print summary information about a Gurobi model."""
    if model is None:
        print("No model available")
        return
        
    try:
        import gurobipy as gp
        from gurobipy import GRB
        
        print(f"\nModel: {model.ModelName}")
        print(f"  Variables: {model.NumVars}")
        print(f"  Constraints: {model.NumConstrs}")
        print(f"  Non-zeros: {model.NumNZs}")
        
        if model.Status == GRB.OPTIMAL:
            print(f"  Status: OPTIMAL")
            print(f"  Objective: {model.ObjVal:.4f}")
        elif model.Status == GRB.TIME_LIMIT:
            print(f"  Status: TIME_LIMIT")
            print(f"  Best Objective: {model.ObjVal:.4f}")
            print(f"  Best Bound: {model.ObjBound:.4f}")
            print(f"  Gap: {model.MIPGap:.4%}")
        else:
            print(f"  Status: {model.Status}")
            
    except Exception as e:
        print(f"Error getting model info: {e}")


def compare_formulations(
    scheduler,
    objectives: List[str] = None
) -> pd.DataFrame:
    """
    Compare performance across different formulation options.
    
    Args:
        scheduler: GroundOpsScheduler instance
        objectives: List of objectives to test
        
    Returns:
        DataFrame with comparison results
    """
    import pandas as pd
    
    if objectives is None:
        objectives = ['minimize_makespan', 'minimize_cost', 'weighted']
        
    results = []
    
    for obj in objectives:
        monitor = PerformanceMonitor()
        monitor.start()
        
        try:
            scheduler.model = None  # Reset
            result = scheduler.optimize(objective=obj)
            monitor.stop()
            
            report = monitor.get_report(scheduler.model, scheduler, obj)
            
            results.append({
                'objective': obj,
                'solve_time': report.solve_stats.solve_time_seconds,
                'variables': report.model_stats.num_variables,
                'constraints': report.model_stats.num_constraints,
                'status': report.solve_stats.status,
                'objective_value': report.solve_stats.obj_value,
                'mip_gap': report.solve_stats.mip_gap,
                'peak_memory_mb': report.memory_stats.peak_memory_mb
            })
        except Exception as e:
            results.append({
                'objective': obj,
                'error': str(e)
            })
            
    return pd.DataFrame(results)
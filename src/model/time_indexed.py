"""
Time-Indexed Formulation for NASA Ground Operations Scheduler.

This module implements a time-indexed MILP formulation that provides
stronger bounds and better handles shift constraints compared to the
disjunctive formulation. Trade-off: larger model size but faster solving
for many practical instances.

Key Features:
- Time-indexed binary variables for task scheduling
- Native support for shift constraints and time windows
- Stronger LP relaxation bounds
- Efficient handling of maintenance windows

Formulation:
- x[t,tau] = 1 if task t starts at time tau
- Resource constraints: sum over tasks active at each time period
- Precedence constraints: sum of weighted start times

Author: Operations Research Portfolio Project
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Warning: Gurobi not available. Install gurobipy for optimization.")


@dataclass
class TimeIndexedConfig:
    """Configuration for time-indexed formulation."""
    time_granularity_hours: float = 1.0  # Time step size
    allow_preemption: bool = False  # Can tasks be interrupted?
    use_discrete_durations: bool = True  # Round durations to time steps
    add_symmetry_breaking: bool = True  # Add symmetry breaking constraints
    add_valid_inequalities: bool = True  # Add cutting planes
    
    
class TimeIndexedScheduler:
    """
    Time-indexed MILP formulation for ground operations scheduling.
    
    This formulation uses binary variables x[t,tau] indicating whether
    task t starts at time period tau. This provides:
    - Stronger LP relaxation bounds
    - Native handling of shift constraints
    - Easy integration of maintenance windows
    - Natural time window constraints
    
    Trade-off: Model size is O(T * H) where T = tasks, H = time periods
    
    Example:
        >>> scheduler = TimeIndexedScheduler(data_dir="data")
        >>> scheduler.load_data()
        >>> result = scheduler.optimize(objective="minimize_makespan")
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        config: TimeIndexedConfig = None
    ):
        """
        Initialize the time-indexed scheduler.
        
        Args:
            data_dir: Path to data directory
            config: Configuration options for the formulation
        """
        self.data_dir = Path(data_dir)
        self.config = config or TimeIndexedConfig()
        
        # Data containers
        self.tasks: Dict = {}
        self.resources: Dict = {}
        self.missions: Dict = {}
        self.constraints_config: Dict = {}
        
        # Time-related attributes
        self.planning_start: datetime = datetime(2028, 1, 1)
        self.planning_end: datetime = datetime(2028, 4, 1)
        self.time_horizon_hours: int = 0
        self.num_periods: int = 0
        self.granularity: float = self.config.time_granularity_hours
        
        # Model components
        self.model: Optional[gp.Model] = None
        self.start_vars: Dict[str, Dict[int, gp.Var]] = {}  # task_id -> {period -> var}
        self.makespan_var: Optional[gp.Var] = None
        self.completion_vars: Dict[str, gp.Var] = {}
        
        # Results
        self.result: Optional[Any] = None
        
    def load_data(self) -> None:
        """Load mission, resource, and constraint data."""
        from src.model.scheduler import Task, Resource
        
        # Load missions
        missions_path = self.data_dir / "missions.json"
        with open(missions_path, 'r') as f:
            missions_data = json.load(f)
            
        for mission in missions_data["missions"]:
            mission_id = mission["id"]
            self.missions[mission_id] = mission
            
            launch_start = datetime.fromisoformat(
                mission["launch_window_start"].replace("Z", "+00:00")
            )
            launch_end = datetime.fromisoformat(
                mission["launch_window_end"].replace("Z", "+00:00")
            )
            
            for task_data in mission["tasks"]:
                task = Task(
                    id=task_data["id"],
                    name=task_data["name"],
                    mission_id=mission_id,
                    duration_hours=task_data["duration_hours"],
                    required_resources=task_data["required_resources"],
                    hazard_level=task_data.get("hazard_level", "low"),
                    predecessors=task_data.get("predecessors", []),
                    safety_buffer_after_hours=task_data.get("safety_buffer_after_hours", 0),
                    launch_window_start=launch_start,
                    launch_window_end=launch_end
                )
                self.tasks[task.id] = task
                
        # Load resources
        resources_path = self.data_dir / "resources.json"
        with open(resources_path, 'r') as f:
            resources_data = json.load(f)
            
        for facility in resources_data.get("facilities", []):
            self.resources[facility["id"]] = Resource(
                id=facility["id"],
                name=facility["name"],
                type="facility",
                capacity=facility["capacity"],
                hourly_cost=facility["hourly_cost"]
            )
            
        for crew in resources_data.get("crews", []):
            self.resources[crew["id"]] = Resource(
                id=crew["id"],
                name=crew["name"],
                type="crew",
                capacity=crew["capacity"],
                hourly_cost=crew["hourly_cost"]
            )
            
        for equipment in resources_data.get("equipment", []):
            self.resources[equipment["id"]] = Resource(
                id=equipment["id"],
                name=equipment["name"],
                type="equipment",
                capacity=equipment["capacity"],
                hourly_cost=equipment["hourly_cost"]
            )
            
        # Load constraints
        constraints_path = self.data_dir / "constraints.json"
        with open(constraints_path, 'r') as f:
            self.constraints_config = json.load(f)
            
        # Set planning horizon
        planning = resources_data.get("planning_horizon", {})
        self.planning_start = datetime.fromisoformat(
            planning.get("start_date", "2028-01-01T00:00:00Z").replace("Z", "+00:00")
        )
        self.planning_end = datetime.fromisoformat(
            planning.get("end_date", "2028-04-01T00:00:00Z").replace("Z", "+00:00")
        )
        self.time_horizon_hours = int(
            (self.planning_end - self.planning_start).total_seconds() / 3600
        )
        self.num_periods = int(self.time_horizon_hours / self.granularity)
        
    def _discretize_duration(self, duration_hours: float) -> int:
        """Convert duration to number of time periods."""
        return max(1, int(np.ceil(duration_hours / self.granularity)))
        
    def _hours_to_period(self, hours: float) -> int:
        """Convert hours to time period index."""
        return int(hours / self.granularity)
        
    def _period_to_hours(self, period: int) -> float:
        """Convert time period to hours."""
        return period * self.granularity
        
    def build_model(self, objective: str = "minimize_makespan") -> None:
        """
        Build the time-indexed MILP model.
        
        Args:
            objective: Optimization objective
        """
        if not GUROBI_AVAILABLE:
            raise RuntimeError("Gurobi is required for optimization.")
            
        self.model = gp.Model("NASA_Ground_Ops_TimeIndexed")
        
        # Set parameters
        opt_settings = self.constraints_config.get("optimization_settings", {})
        self.model.setParam("TimeLimit", opt_settings.get("time_limit_seconds", 3600))
        self.model.setParam("MIPGap", opt_settings.get("mip_gap_tolerance", 0.01))
        self.model.setParam("OutputFlag", 1)
        
        # Create variables
        self._create_variables()
        
        # Add constraints
        self._add_assignment_constraints()
        self._add_precedence_constraints()
        self._add_time_window_constraints()
        self._add_resource_capacity_constraints()
        
        if self.config.add_symmetry_breaking:
            self._add_symmetry_breaking()
            
        if self.config.add_valid_inequalities:
            self._add_valid_inequalities()
            
        # Set objective
        self._set_objective(objective)
        
    def _create_variables(self) -> None:
        """Create time-indexed binary variables."""
        # Binary variables: x[t,tau] = 1 if task t starts at period tau
        for task_id, task in self.tasks.items():
            duration_periods = self._discretize_duration(task.duration_hours)
            latest_start = self.num_periods - duration_periods
            
            self.start_vars[task_id] = {}
            
            for tau in range(latest_start + 1):
                self.start_vars[task_id][tau] = self.model.addVar(
                    name=f"x_{task_id}_{tau}",
                    vtype=GRB.BINARY
                )
                
        # Continuous variables for completion times
        for task_id in self.tasks:
            self.completion_vars[task_id] = self.model.addVar(
                name=f"completion_{task_id}",
                lb=0,
                ub=self.num_periods,
                vtype=GRB.CONTINUOUS
            )
            
        # Makespan variable
        self.makespan_var = self.model.addVar(
            name="makespan",
            lb=0,
            ub=self.num_periods,
            vtype=GRB.CONTINUOUS
        )
        
        self.model.update()
        
    def _add_assignment_constraints(self) -> None:
        """Each task must start exactly once."""
        for task_id, var_dict in self.start_vars.items():
            self.model.addConstr(
                gp.quicksum(var_dict.values()) == 1,
                name=f"assign_{task_id}"
            )
            
        # Link completion times to start times
        for task_id, task in self.tasks.items():
            duration_periods = self._discretize_duration(task.duration_hours)
            
            # completion[t] = sum_tau (tau + duration) * x[t,tau]
            completion_expr = gp.quicksum(
                (tau + duration_periods) * self.start_vars[task_id][tau]
                for tau in self.start_vars[task_id]
            )
            
            self.model.addConstr(
                self.completion_vars[task_id] == completion_expr,
                name=f"completion_def_{task_id}"
            )
            
    def _add_precedence_constraints(self) -> None:
        """Add precedence constraints between tasks."""
        for task_id, task in self.tasks.items():
            for pred_id in task.predecessors:
                if pred_id not in self.tasks:
                    continue
                    
                pred_task = self.tasks[pred_id]
                buffer_periods = self._discretize_duration(
                    pred_task.safety_buffer_after_hours
                )
                pred_duration = self._discretize_duration(pred_task.duration_hours)
                
                # For each possible start time of successor
                for tau in self.start_vars[task_id]:
                    # Task can start at tau only if predecessor completed
                    # tau >= completion[pred] + buffer
                    # Equivalently: tau * x[t,tau] <= sum_tau' (tau' + dur + buf) * x[pred,tau']
                    
                    # Linearized: x[t,tau] <= sum_{tau' <= tau - dur - buf} x[pred,tau']
                    earliest_pred_completion = tau - pred_duration - buffer_periods
                    
                    if earliest_pred_completion >= 0:
                        valid_starts = [
                            self.start_vars[pred_id][tau_p]
                            for tau_p in self.start_vars[pred_id]
                            if tau_p + pred_duration + buffer_periods <= tau
                        ]
                        if valid_starts:
                            self.model.addConstr(
                                self.start_vars[task_id][tau] <= gp.quicksum(valid_starts),
                                name=f"prec_{pred_id}_{task_id}_{tau}"
                            )
                    else:
                        # This start time is too early - must be 0
                        self.model.addConstr(
                            self.start_vars[task_id][tau] == 0,
                            name=f"prec_early_{pred_id}_{task_id}_{tau}"
                        )
                        
    def _add_time_window_constraints(self) -> None:
        """Add time window constraints from launch windows."""
        for task_id, task in self.tasks.items():
            if not task.launch_window_start or not task.launch_window_end:
                continue
                
            window_start_hours = (
                task.launch_window_start - self.planning_start
            ).total_seconds() / 3600
            window_end_hours = (
                task.launch_window_end - self.planning_start
            ).total_seconds() / 3600
            
            window_start_period = self._hours_to_period(window_start_hours)
            window_end_period = self._hours_to_period(window_end_hours)
            duration_periods = self._discretize_duration(task.duration_hours)
            
            if "Launch" in task.name:
                # Launch must start within window
                for tau in self.start_vars[task_id]:
                    if tau < window_start_period or tau > window_end_period - duration_periods:
                        self.model.addConstr(
                            self.start_vars[task_id][tau] == 0,
                            name=f"tw_launch_{task_id}_{tau}"
                        )
            else:
                # Other tasks must complete before window ends
                for tau in self.start_vars[task_id]:
                    if tau + duration_periods > window_end_period:
                        self.model.addConstr(
                            self.start_vars[task_id][tau] == 0,
                            name=f"tw_complete_{task_id}_{tau}"
                        )
                        
    def _add_resource_capacity_constraints(self) -> None:
        """Add resource capacity constraints at each time period."""
        # Group tasks by resource
        resource_to_tasks: Dict[str, List[str]] = {}
        for task_id, task in self.tasks.items():
            for res_id in task.required_resources:
                if res_id not in resource_to_tasks:
                    resource_to_tasks[res_id] = []
                resource_to_tasks[res_id].append(task_id)
                
        # For each resource and time period
        for res_id, task_ids in resource_to_tasks.items():
            if res_id not in self.resources:
                continue
                
            capacity = self.resources[res_id].capacity
            
            for tau in range(self.num_periods):
                # Find all tasks that could be active at time tau
                active_exprs = []
                
                for task_id in task_ids:
                    task = self.tasks[task_id]
                    duration_periods = self._discretize_duration(task.duration_hours)
                    
                    # Task t is active at tau if it started at tau' where
                    # tau' <= tau < tau' + duration
                    for start_tau in self.start_vars[task_id]:
                        if start_tau <= tau < start_tau + duration_periods:
                            active_exprs.append(self.start_vars[task_id][start_tau])
                            
                if active_exprs:
                    self.model.addConstr(
                        gp.quicksum(active_exprs) <= capacity,
                        name=f"cap_{res_id}_{tau}"
                    )
                    
    def _add_symmetry_breaking(self) -> None:
        """Add symmetry breaking constraints for tasks in same mission."""
        # Order tasks within a mission by their IDs when they don't have
        # precedence relationships
        for mission_id in self.missions:
            mission_tasks = [
                (tid, t) for tid, t in self.tasks.items()
                if t.mission_id == mission_id
            ]
            
            # Sort by task ID
            mission_tasks.sort(key=lambda x: x[0])
            
            for i in range(len(mission_tasks) - 1):
                t1_id, t1 = mission_tasks[i]
                t2_id, t2 = mission_tasks[i + 1]
                
                # If no precedence between them, order by completion time
                if t2_id not in t1.predecessors and t1_id not in t2.predecessors:
                    self.model.addConstr(
                        self.completion_vars[t1_id] <= self.completion_vars[t2_id],
                        name=f"symbreak_{t1_id}_{t2_id}"
                    )
                    
    def _add_valid_inequalities(self) -> None:
        """Add valid inequalities (cutting planes) to strengthen formulation."""
        # Energy-based valid inequalities
        # For each resource, total work must fit within time horizon
        resource_to_tasks: Dict[str, List[str]] = {}
        for task_id, task in self.tasks.items():
            for res_id in task.required_resources:
                if res_id not in resource_to_tasks:
                    resource_to_tasks[res_id] = []
                resource_to_tasks[res_id].append(task_id)
                
        for res_id, task_ids in resource_to_tasks.items():
            if res_id not in self.resources:
                continue
                
            capacity = self.resources[res_id].capacity
            
            # Total energy = sum of task durations using this resource
            total_energy = sum(
                self._discretize_duration(self.tasks[tid].duration_hours)
                for tid in task_ids
            )
            
            # Minimum makespan from this resource = total_energy / capacity
            min_makespan = total_energy / capacity
            
            self.model.addConstr(
                self.makespan_var >= min_makespan,
                name=f"energy_cut_{res_id}"
            )
            
    def _set_objective(self, objective: str) -> None:
        """Set the optimization objective."""
        # Makespan definition
        for task_id in self.tasks:
            self.model.addConstr(
                self.makespan_var >= self.completion_vars[task_id],
                name=f"makespan_def_{task_id}"
            )
            
        if objective == "minimize_makespan":
            self.model.setObjective(self.makespan_var, GRB.MINIMIZE)
            
        elif objective == "minimize_cost":
            # Cost = sum of resource costs weighted by duration
            cost_expr = gp.quicksum(
                self.tasks[task_id].duration_hours * 
                sum(
                    self.resources[res_id].hourly_cost
                    for res_id in self.tasks[task_id].required_resources
                    if res_id in self.resources
                ) * gp.quicksum(self.start_vars[task_id].values())
                for task_id in self.tasks
            )
            self.model.setObjective(cost_expr, GRB.MINIMIZE)
            
        elif objective == "weighted":
            # Weighted combination
            alpha = 0.7
            
            cost_expr = gp.quicksum(
                self.tasks[task_id].duration_hours * 
                sum(
                    self.resources[res_id].hourly_cost
                    for res_id in self.tasks[task_id].required_resources
                    if res_id in self.resources
                )
                for task_id in self.tasks
            )
            
            self.model.setObjective(
                alpha * self.makespan_var / self.num_periods + 
                (1 - alpha) * cost_expr / 100000,
                GRB.MINIMIZE
            )
        else:
            self.model.setObjective(self.makespan_var, GRB.MINIMIZE)
            
        self.model.update()
        
    def optimize(self, objective: str = "minimize_makespan") -> Any:
        """
        Solve the optimization problem.
        
        Args:
            objective: Optimization objective
            
        Returns:
            ScheduleResult with the optimal schedule
        """
        if self.model is None:
            self.build_model(objective)
            
        self.model.optimize()
        
        return self._extract_results()
        
    def _extract_results(self) -> Any:
        """Extract solution from solved model."""
        from src.model.scheduler import ScheduleResult
        
        if self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return ScheduleResult(
                status="infeasible",
                makespan_hours=float('inf'),
                total_cost=0,
                tasks=pd.DataFrame(),
                resource_utilization={},
                solve_time_seconds=self.model.Runtime,
                mip_gap=1.0,
                objective_value=0
            )
            
        # Build schedule from solution
        schedule_data = []
        
        for task_id, var_dict in self.start_vars.items():
            task = self.tasks[task_id]
            
            # Find the period where task starts
            start_period = None
            for tau, var in var_dict.items():
                if var.X > 0.5:
                    start_period = tau
                    break
                    
            if start_period is None:
                continue
                
            start_hours = self._period_to_hours(start_period)
            start_time = self.planning_start + timedelta(hours=start_hours)
            end_time = start_time + timedelta(hours=task.duration_hours)
            
            schedule_data.append({
                "task_id": task_id,
                "task_name": task.name,
                "mission_id": task.mission_id,
                "start_time": start_time,
                "end_time": end_time,
                "start_hours": start_hours,
                "duration_hours": task.duration_hours,
                "resources": task.required_resources,
                "hazard_level": task.hazard_level
            })
            
        df = pd.DataFrame(schedule_data).sort_values("start_hours")
        
        # Calculate metrics
        makespan = self.makespan_var.X * self.granularity
        
        # Resource utilization
        resource_utilization = {}
        if len(df) > 0:
            for res_id in self.resources:
                tasks_using = df[
                    df["resources"].apply(lambda x: res_id in x)
                ]
                if len(tasks_using) > 0:
                    busy_hours = tasks_using["duration_hours"].sum()
                    util = (busy_hours / makespan) * 100 if makespan > 0 else 0
                    resource_utilization[res_id] = util
                else:
                    resource_utilization[res_id] = 0
                    
        # Total cost
        total_cost = sum(
            task.duration_hours * sum(
                self.resources[res_id].hourly_cost
                for res_id in task.required_resources
                if res_id in self.resources
            )
            for task in self.tasks.values()
        )
        
        return ScheduleResult(
            status="optimal" if self.model.status == GRB.OPTIMAL else "feasible",
            makespan_hours=makespan,
            total_cost=total_cost,
            tasks=df,
            resource_utilization=resource_utilization,
            solve_time_seconds=self.model.Runtime,
            mip_gap=self.model.MIPGap if hasattr(self.model, 'MIPGap') else 0,
            objective_value=self.model.ObjVal
        )
        
    def add_shift_constraints(
        self,
        shift_starts: List[int],
        shift_ends: List[int]
    ) -> None:
        """
        Add shift constraints - tasks can only start during shifts.
        
        Args:
            shift_starts: List of shift start hours (from planning start)
            shift_ends: List of shift end hours
        """
        for task_id, var_dict in self.start_vars.items():
            for tau in var_dict:
                hours = self._period_to_hours(tau)
                
                # Check if this time falls within any shift
                in_shift = any(
                    start <= hours < end
                    for start, end in zip(shift_starts, shift_ends)
                )
                
                if not in_shift:
                    self.model.addConstr(
                        var_dict[tau] == 0,
                        name=f"shift_{task_id}_{tau}"
                    )
                    
    def add_maintenance_window(
        self,
        resource_id: str,
        start_hour: int,
        end_hour: int
    ) -> None:
        """
        Add maintenance window constraint for a resource.
        
        Args:
            resource_id: Resource under maintenance
            start_hour: Maintenance start (hours from planning start)
            end_hour: Maintenance end
        """
        if resource_id not in self.resources:
            return
            
        start_period = self._hours_to_period(start_hour)
        end_period = self._hours_to_period(end_hour)
        
        # No task using this resource can be active during maintenance
        for task_id, task in self.tasks.items():
            if resource_id not in task.required_resources:
                continue
                
            duration_periods = self._discretize_duration(task.duration_hours)
            
            for tau in self.start_vars[task_id]:
                # Task active during maintenance?
                task_end = tau + duration_periods
                
                if tau < end_period and task_end > start_period:
                    # Overlaps with maintenance - not allowed
                    self.model.addConstr(
                        self.start_vars[task_id][tau] == 0,
                        name=f"maint_{resource_id}_{task_id}_{tau}"
                    )


def compare_formulations(
    data_dir: str = "data"
) -> pd.DataFrame:
    """
    Compare time-indexed vs disjunctive formulations.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame comparing the two approaches
    """
    from src.model.scheduler import GroundOpsScheduler
    
    results = []
    
    # Time-indexed formulation
    try:
        ti_scheduler = TimeIndexedScheduler(data_dir=data_dir)
        ti_scheduler.load_data()
        ti_result = ti_scheduler.optimize()
        
        results.append({
            'formulation': 'time_indexed',
            'solve_time': ti_result.solve_time_seconds,
            'makespan': ti_result.makespan_hours,
            'status': ti_result.status,
            'mip_gap': ti_result.mip_gap
        })
    except Exception as e:
        results.append({
            'formulation': 'time_indexed',
            'error': str(e)
        })
        
    # Disjunctive formulation
    try:
        dj_scheduler = GroundOpsScheduler(data_dir=data_dir)
        dj_scheduler.load_data()
        dj_result = dj_scheduler.optimize()
        
        results.append({
            'formulation': 'disjunctive',
            'solve_time': dj_result.solve_time_seconds,
            'makespan': dj_result.makespan_hours,
            'status': dj_result.status,
            'mip_gap': dj_result.mip_gap
        })
    except Exception as e:
        results.append({
            'formulation': 'disjunctive',
            'error': str(e)
        })
        
    return pd.DataFrame(results)
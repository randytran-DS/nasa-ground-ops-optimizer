"""
Main MILP Scheduler for NASA Ground Operations.

This module implements a Mixed-Integer Linear Programming model for scheduling
ground operations tasks (assembly, testing, fueling, launch) across limited
resources (facilities, crews, equipment) while respecting various constraints.

Model Components:
- Decision Variables: Task start times, resource assignments, sequencing
- Constraints: Precedence, resource capacity, time windows, safety buffers
- Objectives: Minimize makespan, minimize cost, maximize utilization

Author: Operations Research Portfolio Project
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Warning: Gurobi not available. Install gurobipy to use optimization features.")


@dataclass
class Task:
    """Represents a scheduling task."""
    id: str
    name: str
    mission_id: str
    duration_hours: float
    required_resources: List[str]
    hazard_level: str = "low"
    predecessors: List[str] = field(default_factory=list)
    safety_buffer_after_hours: float = 0
    launch_window_start: Optional[datetime] = None
    launch_window_end: Optional[datetime] = None
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class Resource:
    """Represents a schedulable resource."""
    id: str
    name: str
    type: str
    capacity: int
    hourly_cost: float
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class ScheduleResult:
    """Container for optimization results."""
    status: str
    makespan_hours: float
    total_cost: float
    tasks: pd.DataFrame
    resource_utilization: Dict[str, float]
    solve_time_seconds: float
    mip_gap: float
    objective_value: float


class GroundOpsScheduler:
    """
    MILP-based scheduler for NASA ground operations.
    
    This class formulates and solves a mixed-integer linear program to
    optimally schedule ground processing activities for space missions.
    
    Example:
        >>> scheduler = GroundOpsScheduler(data_dir="data")
        >>> scheduler.load_data()
        >>> result = scheduler.optimize(objective="minimize_makespan")
        >>> print(f"Optimal makespan: {result.makespan_hours} hours")
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the scheduler.
        
        Args:
            data_dir: Path to directory containing mission/resource/constraint JSON files
        """
        self.data_dir = Path(data_dir)
        self.tasks: Dict[str, Task] = {}
        self.resources: Dict[str, Resource] = {}
        self.missions: Dict[str, Dict] = {}
        self.constraints_config: Dict = {}
        
        # Model components
        self.model: Optional[gp.Model] = None
        self.start_time_vars: Dict[str, gp.Var] = {}
        self.completion_time_var: Optional[gp.Var] = None
        self.total_cost_var: Optional[gp.Var] = None
        
        # Time discretization (hours from planning horizon start)
        self.planning_start: datetime = datetime(2028, 1, 1)
        self.planning_end: datetime = datetime(2028, 4, 1)
        self.time_horizon_hours: int = 0
        
        # Results
        self.result: Optional[ScheduleResult] = None
        
    def load_data(self) -> None:
        """Load mission, resource, and constraint data from JSON files."""
        # Load missions
        missions_path = self.data_dir / "missions.json"
        with open(missions_path, 'r') as f:
            missions_data = json.load(f)
            
        for mission in missions_data["missions"]:
            mission_id = mission["id"]
            self.missions[mission_id] = mission
            
            # Parse launch windows
            launch_start = datetime.fromisoformat(mission["launch_window_start"].replace("Z", "+00:00"))
            launch_end = datetime.fromisoformat(mission["launch_window_end"].replace("Z", "+00:00"))
            
            # Create tasks for this mission
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
            
        # Combine all resource types
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
        
        # Load constraints config
        constraints_path = self.data_dir / "constraints.json"
        with open(constraints_path, 'r') as f:
            self.constraints_config = json.load(f)
            
        # Set planning horizon
        planning = resources_data.get("planning_horizon", {})
        self.planning_start = datetime.fromisoformat(planning.get("start_date", "2028-01-01T00:00:00Z").replace("Z", "+00:00"))
        self.planning_end = datetime.fromisoformat(planning.get("end_date", "2028-04-01T00:00:00Z").replace("Z", "+00:00"))
        self.time_horizon_hours = int((self.planning_end - self.planning_start).total_seconds() / 3600)
        
    def build_model(self, objective: str = "minimize_makespan") -> None:
        """
        Build the MILP optimization model.
        
        Args:
            objective: Optimization objective - one of:
                - "minimize_makespan": Minimize total completion time
                - "minimize_cost": Minimize total operational cost
                - "maximize_utilization": Maximize resource utilization
        """
        if not GUROBI_AVAILABLE:
            raise RuntimeError("Gurobi is required for optimization. Install gurobipy.")
            
        # Create model
        self.model = gp.Model("NASA_Ground_Ops_Scheduler")
        
        # Set model parameters from config
        opt_settings = self.constraints_config.get("optimization_settings", {})
        self.model.setParam("TimeLimit", opt_settings.get("time_limit_seconds", 3600))
        self.model.setParam("MIPGap", opt_settings.get("mip_gap_tolerance", 0.01))
        self.model.setParam("OutputFlag", 1)
        
        # Create decision variables
        self._create_variables()
        
        # Add constraints
        self._add_precedence_constraints()
        self._add_time_window_constraints()
        self._add_safety_buffer_constraints()
        self._add_resource_capacity_constraints()
        
        # Set objective
        self._set_objective(objective)
        
    def _create_variables(self) -> None:
        """Create all decision variables for the model."""
        # Task start times (continuous, in hours from planning start)
        for task_id, task in self.tasks.items():
            self.start_time_vars[task_id] = self.model.addVar(
                name=f"start_{task_id}",
                lb=0,
                ub=self.time_horizon_hours,
                vtype=GRB.CONTINUOUS
            )
        
        # Makespan variable (maximum completion time)
        self.completion_time_var = self.model.addVar(
            name="makespan",
            lb=0,
            ub=self.time_horizon_hours,
            vtype=GRB.CONTINUOUS
        )
        
        # Total cost variable
        self.total_cost_var = self.model.addVar(
            name="total_cost",
            lb=0,
            vtype=GRB.CONTINUOUS
        )
        
        self.model.update()
        
    def _add_precedence_constraints(self) -> None:
        """
        Add precedence constraints: s[t2] >= s[t1] + d[t1] + buffer[t1]
        
        For each task with predecessors, ensure it starts after all
        predecessors complete (including any required safety buffers).
        """
        for task_id, task in self.tasks.items():
            for pred_id in task.predecessors:
                if pred_id in self.tasks:
                    pred_task = self.tasks[pred_id]
                    buffer = pred_task.safety_buffer_after_hours
                    
                    # s[t2] >= s[t1] + d[t1] + buffer
                    self.model.addConstr(
                        self.start_time_vars[task_id] >= 
                        self.start_time_vars[pred_id] + pred_task.duration_hours + buffer,
                        name=f"precedence_{pred_id}_to_{task_id}"
                    )
                    
    def _add_time_window_constraints(self) -> None:
        """
        Add time window constraints based on launch windows.
        
        Launch tasks must occur within their specified launch windows.
        Other tasks should complete before their mission's launch window.
        """
        for task_id, task in self.tasks.items():
            if task.launch_window_start and task.launch_window_end:
                # Convert to hours from planning start
                window_start_hours = (task.launch_window_start - self.planning_start).total_seconds() / 3600
                window_end_hours = (task.launch_window_end - self.planning_start).total_seconds() / 3600
                
                # For launch tasks, constrain start time within window
                if "Launch" in task.name:
                    self.model.addConstr(
                        self.start_time_vars[task_id] >= max(0, window_start_hours),
                        name=f"window_start_{task_id}"
                    )
                    self.model.addConstr(
                        self.start_time_vars[task_id] <= window_end_hours - task.duration_hours,
                        name=f"window_end_{task_id}"
                    )
                else:
                    # Non-launch tasks should complete before launch window ends
                    self.model.addConstr(
                        self.start_time_vars[task_id] + task.duration_hours <= window_end_hours,
                        name=f"complete_by_window_{task_id}"
                    )
                    
    def _add_safety_buffer_constraints(self) -> None:
        """
        Add safety buffer constraints between hazardous operations.
        
        Based on the constraint configuration, add required time gaps
        between tasks of different hazard levels sharing resources.
        """
        if not self.constraints_config.get("safety_buffer_constraints", {}).get("enforced", True):
            return
            
        buffer_rules = self.constraints_config.get("safety_buffer_constraints", {}).get("default_buffer_hours", {})
        
        # Find pairs of tasks sharing resources
        for t1_id, t1 in self.tasks.items():
            for t2_id, t2 in self.tasks.items():
                if t1_id >= t2_id:
                    continue
                    
                # Check if tasks share resources
                shared_resources = set(t1.required_resources) & set(t2.required_resources)
                if not shared_resources:
                    continue
                    
                # Determine buffer based on hazard levels
                buffer_key = f"{t1.hazard_level}_to_{t2.hazard_level}"
                buffer = buffer_rules.get(buffer_key, 0)
                
                if buffer > 0:
                    # Create binary variables for sequencing
                    y = self.model.addVar(
                        name=f"seq_{t1_id}_{t2_id}",
                        vtype=GRB.BINARY
                    )
                    
                    # Big-M for disjunction
                    M = self.time_horizon_hours
                    
                    # Either t2 starts after t1 completes + buffer, or vice versa
                    # s2 >= s1 + d1 + buffer - M*(1-y)
                    # s1 >= s2 + d2 + buffer - M*y
                    self.model.addConstr(
                        self.start_time_vars[t2_id] >= 
                        self.start_time_vars[t1_id] + t1.duration_hours + buffer - M * (1 - y),
                        name=f"safety_buffer_{t1_id}_before_{t2_id}"
                    )
                    self.model.addConstr(
                        self.start_time_vars[t1_id] >= 
                        self.start_time_vars[t2_id] + t2.duration_hours + buffer - M * y,
                        name=f"safety_buffer_{t2_id}_before_{t1_id}"
                    )
                    
    def _add_resource_capacity_constraints(self) -> None:
        """
        Add resource capacity constraints using time-indexed formulation.
        
        For each time period, the number of tasks using a resource
        cannot exceed the resource's capacity.
        
        Uses a discrete-time approximation with binary variables.
        """
        # Group tasks by shared resources
        resource_to_tasks: Dict[str, List[str]] = {}
        for task_id, task in self.tasks.items():
            for res_id in task.required_resources:
                if res_id not in resource_to_tasks:
                    resource_to_tasks[res_id] = []
                resource_to_tasks[res_id].append(task_id)
        
        # For each resource with capacity constraints
        for res_id, task_ids in resource_to_tasks.items():
            if res_id not in self.resources:
                continue
                
            resource = self.resources[res_id]
            
            # If only one task can use this resource at a time (capacity=1)
            # or if multiple tasks compete for limited capacity
            if len(task_ids) > resource.capacity:
                # Create sequencing variables for all pairs
                for i, t1_id in enumerate(task_ids):
                    for t2_id in task_ids[i+1:]:
                        # Binary: 1 if t1 comes before t2
                        y = self.model.addVar(
                            name=f"order_{res_id}_{t1_id}_{t2_id}",
                            vtype=GRB.BINARY
                        )
                        
                        t1 = self.tasks[t1_id]
                        t2 = self.tasks[t2_id]
                        M = self.time_horizon_hours
                        
                        # Disjunctive constraints
                        self.model.addConstr(
                            self.start_time_vars[t2_id] >= 
                            self.start_time_vars[t1_id] + t1.duration_hours - M * (1 - y),
                            name=f"disj_{res_id}_{t1_id}_before_{t2_id}"
                        )
                        self.model.addConstr(
                            self.start_time_vars[t1_id] >= 
                            self.start_time_vars[t2_id] + t2.duration_hours - M * y,
                            name=f"disj_{res_id}_{t2_id}_before_{t1_id}"
                        )
                        
    def _set_objective(self, objective: str) -> None:
        """
        Set the optimization objective.
        
        Args:
            objective: One of "minimize_makespan", "minimize_cost", "maximize_utilization"
        """
        # Calculate makespan constraint: makespan >= all task completion times
        for task_id, task in self.tasks.items():
            self.model.addConstr(
                self.completion_time_var >= 
                self.start_time_vars[task_id] + task.duration_hours,
                name=f"makespan_bound_{task_id}"
            )
        
        # Calculate total cost
        cost_expr = gp.quicksum(
            self.start_time_vars[task_id] * 0  # Placeholder - cost calculated differently
            for task_id in self.tasks
        )
        
        # More accurate cost calculation based on resource usage
        resource_cost_expr = gp.quicksum(
            task.duration_hours * sum(
                self.resources[res_id].hourly_cost 
                for res_id in task.required_resources 
                if res_id in self.resources
            )
            for task_id, task in self.tasks.items()
        )
        
        if objective == "minimize_makespan":
            self.model.setObjective(self.completion_time_var, GRB.MINIMIZE)
            
        elif objective == "minimize_cost":
            self.model.setObjective(resource_cost_expr, GRB.MINIMIZE)
            
        elif objective == "maximize_utilization":
            # Minimize idle time (equivalent to maximizing utilization)
            # Utilization = total task hours / (makespan * num_resources)
            self.model.setObjective(self.completion_time_var, GRB.MINIMIZE)
            
        elif objective == "weighted":
            # Weighted combination of makespan and cost
            alpha = 0.7  # Weight for makespan
            beta = 0.3   # Weight for cost
            # Normalize makespan and cost for weighting
            self.model.setObjective(
                alpha * self.completion_time_var / 1000 + beta * resource_cost_expr / 100000,
                GRB.MINIMIZE
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")
            
        self.model.update()
        
    def optimize(self, objective: str = "minimize_makespan") -> ScheduleResult:
        """
        Solve the optimization problem.
        
        Args:
            objective: Optimization objective
            
        Returns:
            ScheduleResult containing the optimal schedule
        """
        if self.model is None:
            self.build_model(objective)
            
        # Solve
        self.model.optimize()
        
        # Extract results
        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.TIME_LIMIT:
            return self._extract_results()
        else:
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
            
    def _extract_results(self) -> ScheduleResult:
        """Extract solution from solved model."""
        # Build task schedule DataFrame
        schedule_data = []
        for task_id, task in self.tasks.items():
            start_hours = self.start_time_vars[task_id].X
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
        
        # Calculate makespan
        makespan = self.completion_time_var.X
        
        # Calculate resource utilization
        resource_utilization = self._calculate_utilization(df)
        
        # Calculate total cost
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
        
    def _calculate_utilization(self, schedule_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate resource utilization percentages."""
        utilization = {}
        
        for res_id in self.resources:
            # Find all tasks using this resource
            tasks_using = schedule_df[
                schedule_df["resources"].apply(lambda x: res_id in x)
            ]
            
            if len(tasks_using) == 0:
                utilization[res_id] = 0.0
                continue
                
            # Sum up task durations
            total_busy_hours = tasks_using["duration_hours"].sum()
            
            # Calculate utilization
            makespan = schedule_df["start_hours"].max() + schedule_df["duration_hours"].max() - schedule_df["start_hours"].min()
            if makespan > 0:
                utilization[res_id] = (total_busy_hours / makespan) * 100
            else:
                utilization[res_id] = 0.0
                
        return utilization
        
    def get_critical_path(self) -> List[str]:
        """
        Identify the critical path through the schedule.
        
        Returns:
            List of task IDs on the critical path
        """
        if self.result is None:
            return []
            
        # Build dependency graph and compute longest path
        # This is a simplified version - full implementation would use
        # dual values or explicit critical path computation
        
        df = self.result.tasks
        if len(df) == 0:
            return []
            
        # Sort by start time and find path with minimal slack
        sorted_tasks = df.sort_values("start_hours")
        
        critical_path = []
        current_time = 0
        
        for _, row in sorted_tasks.iterrows():
            slack = row["start_hours"] - current_time
            if slack < 0.1:  # Essentially zero slack
                critical_path.append(row["task_id"])
                current_time = row["start_hours"] + row["duration_hours"]
                
        return critical_path
        
    def export_schedule(self, output_path: str) -> None:
        """
        Export the schedule to a CSV file.
        
        Args:
            output_path: Path to save the CSV file
        """
        if self.result is None:
            raise ValueError("No schedule to export. Run optimize() first.")
            
        self.result.tasks.to_csv(output_path, index=False)
        
    def print_summary(self) -> None:
        """Print a summary of the optimization results."""
        if self.result is None:
            print("No results available. Run optimize() first.")
            return
            
        print("\n" + "="*60)
        print("NASA GROUND OPERATIONS SCHEDULE - OPTIMIZATION RESULTS")
        print("="*60)
        print(f"\nStatus: {self.result.status.upper()}")
        print(f"Solve Time: {self.result.solve_time_seconds:.2f} seconds")
        print(f"MIP Gap: {self.result.mip_gap:.4%}")
        print(f"\nMakespan: {self.result.makespan_hours:.1f} hours ({self.result.makespan_hours/24:.1f} days)")
        print(f"Total Cost: ${self.result.total_cost:,.2f}")
        
        print("\n" + "-"*60)
        print("SCHEDULE BY MISSION:")
        print("-"*60)
        
        for mission_id in self.result.tasks["mission_id"].unique():
            mission_df = self.result.tasks[self.result.tasks["mission_id"] == mission_id]
            mission_info = self.missions.get(mission_id, {})
            print(f"\n{mission_info.get('name', mission_id)}:")
            
            for _, row in mission_df.iterrows():
                print(f"  {row['start_time'].strftime('%Y-%m-%d %H:%M')} - {row['task_name']} ({row['duration_hours']}h)")
                
        print("\n" + "-"*60)
        print("RESOURCE UTILIZATION:")
        print("-"*60)
        
        for res_id, util in sorted(self.result.resource_utilization.items(), key=lambda x: -x[1]):
            res_name = self.resources[res_id].name if res_id in self.resources else res_id
            print(f"  {res_name}: {util:.1f}%")
            
        critical_path = self.get_critical_path()
        if critical_path:
            print("\n" + "-"*60)
            print("CRITICAL PATH:")
            print("-"*60)
            for task_id in critical_path:
                task = self.tasks[task_id]
                print(f"  → {task.name}")
                
        print("\n" + "="*60)


def main():
    """Main entry point for the scheduler."""
    import argparse
    
    parser = argparse.ArgumentParser(description="NASA Ground Operations Scheduler")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--objective", default="minimize_makespan", 
                       choices=["minimize_makespan", "minimize_cost", "maximize_utilization", "weighted"],
                       help="Optimization objective")
    parser.add_argument("--output", default="schedule.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Create scheduler and run
    scheduler = GroundOpsScheduler(data_dir=args.data_dir)
    scheduler.load_data()
    
    print(f"Loaded {len(scheduler.tasks)} tasks and {len(scheduler.resources)} resources")
    print(f"Planning horizon: {scheduler.planning_start} to {scheduler.planning_end}")
    print(f"Objective: {args.objective}")
    
    result = scheduler.optimize(objective=args.objective)
    
    scheduler.print_summary()
    scheduler.export_schedule(args.output)
    print(f"\nSchedule exported to {args.output}")


if __name__ == "__main__":
    main()
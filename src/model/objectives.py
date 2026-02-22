"""
Objective function definitions for NASA Ground Operations Scheduler.

This module provides different objective functions that can be used
to optimize the ground operations schedule:

1. Minimize Makespan - Complete all tasks as quickly as possible
2. Minimize Cost - Minimize total operational cost
3. Maximize Utilization - Maximize resource usage efficiency
4. Multi-objective - Weighted combination of objectives

Author: Operations Research Portfolio Project
"""

from typing import Dict, List, Optional
from enum import Enum

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class ObjectiveType(Enum):
    """Available objective function types."""
    MINIMIZE_MAKESPAN = "minimize_makespan"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_UTILIZATION = "maximize_utilization"
    MINIMIZE_TARDINESS = "minimize_tardiness"
    WEIGHTED_MULTI = "weighted_multi"


class ObjectiveBuilder:
    """
    Builder class for constructing objective functions.
    
    Provides methods to create and combine different objective
    functions for the scheduling model.
    
    Example:
        >>> builder = ObjectiveBuilder(model, tasks, resources, start_vars)
        >>> builder.set_makespan_objective()
        >>> # Or for multi-objective:
        >>> builder.set_weighted_objective(weights={'makespan': 0.7, 'cost': 0.3})
    """
    
    def __init__(
        self,
        model: "gp.Model",
        tasks: Dict,
        resources: Dict,
        start_vars: Dict[str, "gp.Var"],
        makespan_var: "gp.Var" = None,
        planning_horizon_hours: float = 2160
    ):
        """
        Initialize the objective builder.
        
        Args:
            model: Gurobi model
            tasks: Dictionary of Task objects
            resources: Dictionary of Resource objects
            start_vars: Dictionary mapping task_id to start time variable
            makespan_var: Variable representing makespan
            planning_horizon_hours: Total planning horizon for normalization
        """
        self.model = model
        self.tasks = tasks
        self.resources = resources
        self.start_vars = start_vars
        self.makespan_var = makespan_var
        self.planning_horizon = planning_horizon_hours
        
        # Create makespan variable if not provided
        if self.makespan_var is None:
            self.makespan_var = model.addVar(
                name="makespan",
                lb=0,
                ub=planning_horizon_hours,
                vtype=GRB.CONTINUOUS
            )
            
        # Track objective components
        self.objectives: Dict[str, gp.LinExpr] = {}
        
    def set_makespan_objective(self) -> None:
        """
        Set objective to minimize makespan (total completion time).
        
        Objective: min max_t (start_t + duration_t)
        
        This pushes all tasks to complete as early as possible.
        """
        # Add makespan definition constraints
        for task_id, task in self.tasks.items():
            self.model.addConstr(
                self.makespan_var >= self.start_vars[task_id] + task.duration_hours,
                name=f"makespan_def_{task_id}"
            )
            
        self.model.setObjective(self.makespan_var, GRB.MINIMIZE)
        self.objectives["makespan"] = self.makespan_var
        
    def set_cost_objective(self) -> None:
        """
        Set objective to minimize total operational cost.
        
        Objective: min sum_t (duration_t * sum_r cost_r)
        
        This minimizes the total cost of resource usage.
        """
        cost_expr = gp.quicksum(
            task.duration_hours * sum(
                self.resources[res_id].hourly_cost
                for res_id in task.required_resources
                if res_id in self.resources
            )
            for task_id, task in self.tasks.items()
        )
        
        self.model.setObjective(cost_expr, GRB.MINIMIZE)
        self.objectives["cost"] = cost_expr
        
    def set_utilization_objective(self) -> None:
        """
        Set objective to maximize resource utilization.
        
        This is implemented as minimizing idle time, which is
        approximately equivalent to minimizing makespan for
        fixed work content.
        
        Objective: min makespan (equivalent to max utilization)
        """
        # For fixed total work, minimizing makespan maximizes utilization
        self.set_makespan_objective()
        
    def set_tardiness_objective(
        self,
        deadlines: Dict[str, float],
        tardiness_vars: Dict[str, "gp.Var"] = None
    ) -> None:
        """
        Set objective to minimize total tardiness.
        
        Objective: min sum_t max(0, completion_t - deadline_t)
        
        Args:
            deadlines: Dictionary mapping task_id to deadline (hours from start)
            tardiness_vars: Optional pre-created tardiness variables
        """
        tardiness_exprs = []
        
        for task_id, deadline in deadlines.items():
            if task_id not in self.tasks:
                continue
                
            task = self.tasks[task_id]
            
            # Create tardiness variable
            if tardiness_vars and task_id in tardiness_vars:
                tard_var = tardiness_vars[task_id]
            else:
                tard_var = self.model.addVar(
                    name=f"tardiness_{task_id}",
                    lb=0,
                    vtype=GRB.CONTINUOUS
                )
                
            # Tardiness = max(0, completion - deadline)
            completion = self.start_vars[task_id] + task.duration_hours
            self.model.addConstr(
                tard_var >= completion - deadline,
                name=f"tardiness_def_{task_id}"
            )
            
            tardiness_exprs.append(tard_var)
            
        total_tardiness = gp.quicksum(tardiness_exprs)
        self.model.setObjective(total_tardiness, GRB.MINIMIZE)
        self.objectives["tardiness"] = total_tardiness
        
    def set_weighted_objective(
        self,
        weights: Dict[str, float] = None,
        normalize: bool = True
    ) -> None:
        """
        Set a weighted multi-objective function.
        
        Objective: min w1*makespan/norm1 + w2*cost/norm2 + ...
        
        Args:
            weights: Dictionary with weights for each objective component
                     Default: {'makespan': 0.6, 'cost': 0.4}
            normalize: Whether to normalize objectives to similar scales
        """
        if weights is None:
            weights = {'makespan': 0.6, 'cost': 0.4}
            
        objective_components = []
        
        # Makespan component
        if 'makespan' in weights and weights['makespan'] > 0:
            # Add makespan constraints
            for task_id, task in self.tasks.items():
                self.model.addConstr(
                    self.makespan_var >= self.start_vars[task_id] + task.duration_hours,
                    name=f"makespan_def_{task_id}"
                )
                
            if normalize:
                # Normalize by planning horizon
                norm_factor = self.planning_horizon
                objective_components.append(
                    weights['makespan'] * self.makespan_var / norm_factor
                )
            else:
                objective_components.append(weights['makespan'] * self.makespan_var)
                
        # Cost component
        if 'cost' in weights and weights['cost'] > 0:
            total_cost = gp.quicksum(
                task.duration_hours * sum(
                    self.resources[res_id].hourly_cost
                    for res_id in task.required_resources
                    if res_id in self.resources
                )
                for task_id, task in self.tasks.items()
            )
            
            if normalize:
                # Estimate max cost for normalization
                max_cost = sum(
                    task.duration_hours * sum(
                        self.resources[res_id].hourly_cost
                        for res_id in task.required_resources
                        if res_id in self.resources
                    )
                    for task in self.tasks.values()
                )
                objective_components.append(
                    weights['cost'] * total_cost / max_cost if max_cost > 0 else 0
                )
            else:
                objective_components.append(weights['cost'] * total_cost)
                
        # Combine all components
        combined_objective = gp.quicksum(objective_components)
        self.model.setObjective(combined_objective, GRB.MINIMIZE)
        self.objectives["weighted"] = combined_objective
        
    def set_priority_objective(
        self,
        priorities: Dict[str, int]
    ) -> None:
        """
        Set objective that considers mission priorities.
        
        Higher priority missions get scheduled first (earlier completion).
        
        Args:
            priorities: Dictionary mapping mission_id to priority (1=highest)
        """
        completion_exprs = []
        
        for task_id, task in self.tasks.items():
            mission_id = task.mission_id
            priority = priorities.get(mission_id, 10)  # Default low priority
            
            # Weight completion time by priority (lower priority number = higher weight)
            weight = 1.0 / priority
            completion = self.start_vars[task_id] + task.duration_hours
            completion_exprs.append(weight * completion)
            
        weighted_completion = gp.quicksum(completion_exprs)
        self.model.setObjective(weighted_completion, GRB.MINIMIZE)
        self.objectives["priority"] = weighted_completion
        
    def get_objective_info(self) -> Dict:
        """Get information about current objective setup."""
        return {
            "objectives_defined": list(self.objectives.keys()),
            "objective_count": len(self.objectives)
        }


def calculate_schedule_metrics(
    schedule_df,
    resources: Dict,
    planning_start
) -> Dict:
    """
    Calculate various metrics for a given schedule.
    
    Args:
        schedule_df: DataFrame with schedule results
        resources: Dictionary of Resource objects
        planning_start: Start datetime of planning horizon
        
    Returns:
        Dictionary with calculated metrics
    """
    import pandas as pd
    import numpy as np
    
    metrics = {}
    
    if len(schedule_df) == 0:
        return metrics
        
    # Makespan
    metrics['makespan_hours'] = (
        schedule_df['start_hours'].max() + 
        schedule_df['duration_hours'].max() - 
        schedule_df['start_hours'].min()
    )
    
    # Total task hours
    metrics['total_task_hours'] = schedule_df['duration_hours'].sum()
    
    # Average task duration
    metrics['avg_task_duration'] = schedule_df['duration_hours'].mean()
    
    # Resource utilization
    utilization = {}
    for res_id in resources:
        tasks_using = schedule_df[
            schedule_df['resources'].apply(lambda x: res_id in x)
        ]
        if len(tasks_using) > 0:
            busy_hours = tasks_using['duration_hours'].sum()
            util = (busy_hours / metrics['makespan_hours']) * 100 if metrics['makespan_hours'] > 0 else 0
            utilization[res_id] = util
        else:
            utilization[res_id] = 0
            
    metrics['resource_utilization'] = utilization
    metrics['avg_utilization'] = np.mean(list(utilization.values()))
    
    # Cost calculation
    total_cost = 0
    for _, row in schedule_df.iterrows():
        task_cost = sum(
            resources[res_id].hourly_cost
            for res_id in row['resources']
            if res_id in resources
        )
        total_cost += row['duration_hours'] * task_cost
        
    metrics['total_cost'] = total_cost
    
    # Time distribution
    metrics['earliest_start'] = schedule_df['start_time'].min()
    metrics['latest_end'] = schedule_df['end_time'].max()
    
    return metrics


def compare_objectives(
    scheduler,
    objectives: List[str] = None
) -> Dict:
    """
    Compare results across different objective functions.
    
    Args:
        scheduler: GroundOpsScheduler instance with data loaded
        objectives: List of objective types to compare
        
    Returns:
        Dictionary with comparison results
    """
    if objectives is None:
        objectives = [
            'minimize_makespan',
            'minimize_cost',
            'maximize_utilization',
            'weighted'
        ]
        
    results = {}
    
    for obj in objectives:
        try:
            scheduler.model = None  # Reset model
            result = scheduler.optimize(objective=obj)
            results[obj] = {
                'status': result.status,
                'makespan': result.makespan_hours,
                'cost': result.total_cost,
                'solve_time': result.solve_time_seconds
            }
        except Exception as e:
            results[obj] = {'error': str(e)}
            
    return results
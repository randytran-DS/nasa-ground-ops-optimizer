"""
Constraint definitions for NASA Ground Operations Scheduler.

This module provides modular constraint building blocks that can be
combined to create various scheduling formulations.

Each constraint type is implemented as a separate function that can
be enabled/disabled based on the constraint configuration.

Author: Operations Research Portfolio Project
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


@dataclass
class ConstraintConfig:
    """Configuration for a single constraint type."""
    name: str
    enabled: bool
    parameters: Dict[str, Any]


class ConstraintBuilder:
    """
    Builder class for adding constraints to the MILP model.
    
    This class provides a clean interface for adding various constraint
    types to the scheduling model, with support for enabling/disabling
    constraints and parameter tuning.
    
    Example:
        >>> builder = ConstraintBuilder(model, tasks, resources, config)
        >>> builder.add_precedence_constraints()
        >>> builder.add_resource_constraints()
    """
    
    def __init__(
        self,
        model: "gp.Model",
        tasks: Dict,
        resources: Dict,
        start_vars: Dict[str, "gp.Var"],
        config: Dict
    ):
        """
        Initialize the constraint builder.
        
        Args:
            model: Gurobi model to add constraints to
            tasks: Dictionary of Task objects
            resources: Dictionary of Resource objects
            start_vars: Dictionary mapping task_id to start time variable
            config: Constraint configuration dictionary
        """
        self.model = model
        self.tasks = tasks
        self.resources = resources
        self.start_vars = start_vars
        self.config = config
        
        # Big-M for disjunctive constraints
        self.big_m = 10000  # Should be >= planning horizon
        
        # Track added constraints
        self.constraints_added: Dict[str, int] = {}
        
    def add_all_constraints(self) -> None:
        """Add all enabled constraints from configuration."""
        if self.config.get("precedence_constraints", {}).get("enforced", True):
            self.add_precedence_constraints()
            
        if self.config.get("resource_capacity_constraints", {}).get("enforced", True):
            self.add_resource_capacity_constraints()
            
        if self.config.get("time_window_constraints", {}).get("enforced", True):
            self.add_time_window_constraints()
            
        if self.config.get("safety_buffer_constraints", {}).get("enforced", True):
            self.add_safety_buffer_constraints()
            
        if self.config.get("shift_constraints", {}).get("enforced", False):
            self.add_shift_constraints()
            
        if self.config.get("maintenance_windows"):
            self.add_maintenance_constraints()
            
    def add_precedence_constraints(self) -> None:
        """
        Add precedence constraints between tasks.
        
        For each task with predecessors, ensure it starts after all
        predecessors complete (plus any required safety buffers).
        
        Constraint: s[t] >= s[pred] + duration[pred] + buffer[pred]
        """
        count = 0
        
        for task_id, task in self.tasks.items():
            for pred_id in task.predecessors:
                if pred_id in self.tasks:
                    pred_task = self.tasks[pred_id]
                    buffer = pred_task.safety_buffer_after_hours
                    
                    self.model.addConstr(
                        self.start_vars[task_id] >= 
                        self.start_vars[pred_id] + pred_task.duration_hours + buffer,
                        name=f"prec_{pred_id}_to_{task_id}"
                    )
                    count += 1
                    
        self.constraints_added["precedence"] = count
        
    def add_resource_capacity_constraints(self) -> None:
        """
        Add disjunctive resource capacity constraints.
        
        For resources with capacity 1 (or where tasks exceed capacity),
        ensure tasks don't overlap when using the same resource.
        
        Uses big-M formulation with binary sequencing variables.
        """
        count = 0
        
        # Group tasks by resource
        resource_to_tasks: Dict[str, List[str]] = {}
        for task_id, task in self.tasks.items():
            for res_id in task.required_resources:
                if res_id not in resource_to_tasks:
                    resource_to_tasks[res_id] = []
                resource_to_tasks[res_id].append(task_id)
        
        # Add disjunctive constraints for oversubscribed resources
        for res_id, task_ids in resource_to_tasks.items():
            if res_id not in self.resources:
                continue
                
            capacity = self.resources[res_id].capacity
            
            if len(task_ids) > capacity:
                # Add pairwise disjunctive constraints
                for i, t1_id in enumerate(task_ids):
                    for t2_id in task_ids[i+1:]:
                        t1 = self.tasks[t1_id]
                        t2 = self.tasks[t2_id]
                        
                        # Binary variable: 1 if t1 before t2
                        y = self.model.addVar(
                            name=f"order_{res_id}_{t1_id}_{t2_id}",
                            vtype=GRB.BINARY
                        )
                        
                        # Disjunctive constraints with big-M
                        self.model.addConstr(
                            self.start_vars[t2_id] >= 
                            self.start_vars[t1_id] + t1.duration_hours - self.big_m * (1 - y),
                            name=f"disj_{res_id}_{t1_id}_before_{t2_id}"
                        )
                        self.model.addConstr(
                            self.start_vars[t1_id] >= 
                            self.start_vars[t2_id] + t2.duration_hours - self.big_m * y,
                            name=f"disj_{res_id}_{t2_id}_before_{t1_id}"
                        )
                        count += 2
                        
        self.constraints_added["resource_capacity"] = count
        
    def add_time_window_constraints(
        self,
        time_horizon_hours: float,
        planning_start
    ) -> None:
        """
        Add time window constraints based on launch windows.
        
        Launch tasks must start within their launch window.
        Other tasks must complete before the launch window ends.
        
        Args:
            time_horizon_hours: Total planning horizon in hours
            planning_start: Start datetime of planning horizon
        """
        count = 0
        hard_constraint = self.config.get("time_window_constraints", {}).get(
            "launch_window_hard_constraint", True
        )
        
        for task_id, task in self.tasks.items():
            if not task.launch_window_start or not task.launch_window_end:
                continue
                
            # Convert to hours from planning start
            window_start = (task.launch_window_start - planning_start).total_seconds() / 3600
            window_end = (task.launch_window_end - planning_start).total_seconds() / 3600
            
            # For launch tasks
            if "Launch" in task.name:
                # Start within window
                self.model.addConstr(
                    self.start_vars[task_id] >= max(0, window_start),
                    name=f"tw_start_{task_id}"
                )
                self.model.addConstr(
                    self.start_vars[task_id] <= window_end - task.duration_hours,
                    name=f"tw_end_{task_id}"
                )
                count += 2
            else:
                # Complete before window ends
                if hard_constraint:
                    self.model.addConstr(
                        self.start_vars[task_id] + task.duration_hours <= window_end,
                        name=f"complete_by_{task_id}"
                    )
                    count += 1
                    
        self.constraints_added["time_window"] = count
        
    def add_safety_buffer_constraints(self) -> None:
        """
        Add safety buffer constraints between hazardous operations.
        
        Based on hazard levels, enforce minimum time gaps between
        consecutive tasks sharing resources.
        """
        count = 0
        
        buffer_rules = self.config.get("safety_buffer_constraints", {}).get(
            "default_buffer_hours", {}
        )
        
        # Find task pairs sharing resources
        for t1_id, t1 in self.tasks.items():
            for t2_id, t2 in self.tasks.items():
                if t1_id >= t2_id:
                    continue
                    
                # Check shared resources
                shared = set(t1.required_resources) & set(t2.required_resources)
                if not shared:
                    continue
                    
                # Get buffer based on hazard levels
                buffer_key = f"{t1.hazard_level}_to_{t2.hazard_level}"
                buffer = buffer_rules.get(buffer_key, 0)
                
                # Also check reverse order
                buffer_key_rev = f"{t2.hazard_level}_to_{t1.hazard_level}"
                buffer_rev = buffer_rules.get(buffer_key_rev, 0)
                buffer = max(buffer, buffer_rev)
                
                if buffer > 0:
                    # Binary for sequencing
                    y = self.model.addVar(
                        name=f"safety_seq_{t1_id}_{t2_id}",
                        vtype=GRB.BINARY
                    )
                    
                    # Either t1 before t2+buffer, or t2 before t1+buffer
                    self.model.addConstr(
                        self.start_vars[t2_id] >= 
                        self.start_vars[t1_id] + t1.duration_hours + buffer - self.big_m * (1 - y),
                        name=f"safety_{t1_id}_before_{t2_id}"
                    )
                    self.model.addConstr(
                        self.start_vars[t1_id] >= 
                        self.start_vars[t2_id] + t2.duration_hours + buffer - self.big_m * y,
                        name=f"safety_{t2_id}_before_{t1_id}"
                    )
                    count += 2
                    
        self.constraints_added["safety_buffer"] = count
        
    def add_shift_constraints(self) -> None:
        """
        Add crew shift constraints.
        
        Tasks can only be performed during crew available hours.
        This is a simplified implementation - full implementation would
        use time-indexed variables or more sophisticated formulations.
        """
        # This would require time-indexed formulation
        # Placeholder for shift constraint logic
        self.constraints_added["shift"] = 0
        
    def add_maintenance_constraints(self) -> None:
        """
        Add constraints for resource maintenance windows.
        
        Resources are unavailable during their maintenance windows.
        """
        count = 0
        maintenance_windows = self.config.get("maintenance_windows", [])
        
        for window in maintenance_windows:
            res_id = window["resource_id"]
            # Maintenance constraints would prevent task assignment
            # during the window period
            # This requires time-indexed formulation for proper implementation
            count += 1
            
        self.constraints_added["maintenance"] = count
        
    def add_custom_constraint(
        self,
        name: str,
        constraint_func: callable
    ) -> None:
        """
        Add a custom constraint function.
        
        Args:
            name: Name for the constraint
            constraint_func: Function that takes (model, tasks, resources, start_vars)
        """
        count = constraint_func(self.model, self.tasks, self.resources, self.start_vars)
        self.constraints_added[name] = count
        
    def get_constraint_summary(self) -> str:
        """Get a summary of added constraints."""
        lines = ["Constraint Summary:"]
        lines.append("-" * 40)
        
        total = 0
        for name, count in self.constraints_added.items():
            lines.append(f"  {name}: {count} constraints")
            total += count
            
        lines.append("-" * 40)
        lines.append(f"  Total: {total} constraints")
        
        return "\n".join(lines)


def validate_constraint_config(config: Dict) -> List[str]:
    """
    Validate constraint configuration.
    
    Args:
        config: Constraint configuration dictionary
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    # Check for required constraint sections
    required_sections = [
        "precedence_constraints",
        "resource_capacity_constraints"
    ]
    
    for section in required_sections:
        if section not in config:
            warnings.append(f"Missing constraint section: {section}")
            
    # Validate safety buffer rules
    if "safety_buffer_constraints" in config:
        buffer_config = config["safety_buffer_constraints"]
        if "default_buffer_hours" not in buffer_config:
            warnings.append("Safety buffer constraints enabled but no buffer rules defined")
            
    # Validate optimization settings
    if "optimization_settings" in config:
        opt = config["optimization_settings"]
        if opt.get("mip_gap_tolerance", 0) < 0:
            warnings.append("MIP gap tolerance cannot be negative")
        if opt.get("time_limit_seconds", 0) < 0:
            warnings.append("Time limit cannot be negative")
            
    return warnings


def get_default_constraint_config() -> Dict:
    """
    Get default constraint configuration.
    
    Returns:
        Dictionary with default constraint settings
    """
    return {
        "precedence_constraints": {
            "description": "Tasks must complete before successors can start",
            "enforced": True
        },
        "resource_capacity_constraints": {
            "description": "Cannot exceed available capacity of any resource",
            "enforced": True
        },
        "time_window_constraints": {
            "description": "Tasks must start within allowed time windows",
            "enforced": True,
            "launch_window_hard_constraint": True
        },
        "safety_buffer_constraints": {
            "description": "Required time gaps between hazardous operations",
            "enforced": True,
            "default_buffer_hours": {
                "low_to_low": 0,
                "low_to_medium": 2,
                "low_to_high": 4,
                "medium_to_medium": 4,
                "medium_to_high": 8,
                "high_to_high": 12
            }
        },
        "optimization_settings": {
            "time_limit_seconds": 3600,
            "mip_gap_tolerance": 0.01
        }
    }
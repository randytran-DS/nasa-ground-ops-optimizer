"""
What-If Analysis and Scenario Simulation.

This module provides tools for exploring how changes to input parameters
affect the optimal schedule. This is crucial for:

1. Sensitivity Analysis - How sensitive is the schedule to parameter changes?
2. Risk Assessment - What happens if delays occur?
3. Resource Planning - Do we need more resources?
4. Decision Support - Compare alternative scenarios

Author: Operations Research Portfolio Project
"""

import json
import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import random

import pandas as pd
import numpy as np

# Import scheduler components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.scheduler import GroundOpsScheduler, ScheduleResult


@dataclass
class ScenarioResult:
    """Container for scenario analysis results."""
    scenario_name: str
    base_result: ScheduleResult
    scenario_result: ScheduleResult
    parameter_changes: Dict[str, Any]
    metrics_delta: Dict[str, float]
    
    def summary(self) -> str:
        """Generate a summary string of the scenario comparison."""
        lines = [
            f"\n{'='*60}",
            f"Scenario: {self.scenario_name}",
            f"{'='*60}",
            f"Parameter Changes: {self.parameter_changes}",
            f"",
            f"Base Case:",
            f"  Makespan: {self.base_result.makespan_hours:.1f} hours",
            f"  Cost: ${self.base_result.total_cost:,.2f}",
            f"",
            f"Scenario Result:",
            f"  Makespan: {self.scenario_result.makespan_hours:.1f} hours",
            f"  Cost: ${self.scenario_result.total_cost:,.2f}",
            f"",
            f"Impact:",
        ]
        
        for metric, delta in self.metrics_delta.items():
            sign = "+" if delta > 0 else ""
            lines.append(f"  {metric}: {sign}{delta:.2f}")
            
        return "\n".join(lines)


@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results."""
    parameter_name: str
    parameter_values: List[Any]
    makespan_values: List[float]
    cost_values: List[float]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        return pd.DataFrame({
            self.parameter_name: self.parameter_values,
            'makespan_hours': self.makespan_values,
            'total_cost': self.cost_values
        })


class WhatIfAnalyzer:
    """
    What-If Analysis toolkit for schedule exploration.
    
    Provides methods for:
    - Scenario analysis with parameter modifications
    - Sensitivity analysis across parameter ranges
    - Monte Carlo simulation for uncertainty quantification
    - Resource impact analysis
    
    Example:
        >>> analyzer = WhatIfAnalyzer(base_data_dir="data")
        >>> analyzer.load_base_case()
        >>> result = analyzer.run_scenario({"task_duration_multiplier": 1.2})
    """
    
    def __init__(self, base_data_dir: str = "data"):
        """
        Initialize the what-if analyzer.
        
        Args:
            base_data_dir: Path to base case data directory
        """
        self.base_data_dir = Path(base_data_dir)
        self.base_missions: Dict = {}
        self.base_resources: Dict = {}
        self.base_constraints: Dict = {}
        self.base_result: Optional[ScheduleResult] = None
        
    def load_base_case(self, objective: str = "minimize_makespan") -> ScheduleResult:
        """
        Load and solve the base case scenario.
        
        Args:
            objective: Optimization objective
            
        Returns:
            ScheduleResult for the base case
        """
        # Load base data
        with open(self.base_data_dir / "missions.json", 'r') as f:
            missions_data = json.load(f)
            self.base_missions = missions_data["missions"]
            
        with open(self.base_data_dir / "resources.json", 'r') as f:
            self.base_resources = json.load(f)
            
        with open(self.base_data_dir / "constraints.json", 'r') as f:
            self.base_constraints = json.load(f)
            
        # Solve base case
        scheduler = GroundOpsScheduler(data_dir=str(self.base_data_dir))
        scheduler.load_data()
        self.base_result = scheduler.optimize(objective=objective)
        
        return self.base_result
        
    def run_scenario(
        self,
        scenario_params: Dict[str, Any],
        scenario_name: str = "custom"
    ) -> ScenarioResult:
        """
        Run a what-if scenario with modified parameters.
        
        Args:
            scenario_params: Dictionary of parameter modifications
            scenario_name: Name for the scenario
            
        Returns:
            ScenarioResult comparing base case to scenario
        """
        if self.base_result is None:
            self.load_base_case()
            
        # Create modified data
        modified_missions = copy.deepcopy(self.base_missions)
        modified_resources = copy.deepcopy(self.base_resources)
        modified_constraints = copy.deepcopy(self.base_constraints)
        
        # Apply parameter modifications
        modified_missions, modified_resources, modified_constraints = \
            self._apply_modifications(
                modified_missions, 
                modified_resources, 
                modified_constraints,
                scenario_params
            )
        
        # Create temporary directory with modified data
        temp_dir = self._create_temp_scenario(
            modified_missions,
            modified_resources,
            modified_constraints
        )
        
        # Solve scenario
        scheduler = GroundOpsScheduler(data_dir=str(temp_dir))
        scheduler.load_data()
        scenario_result = scheduler.optimize()
        
        # Calculate deltas
        metrics_delta = {
            'makespan_hours': scenario_result.makespan_hours - self.base_result.makespan_hours,
            'total_cost': scenario_result.total_cost - self.base_result.total_cost,
            'solve_time': scenario_result.solve_time_seconds - self.base_result.solve_time_seconds
        }
        
        return ScenarioResult(
            scenario_name=scenario_name,
            base_result=self.base_result,
            scenario_result=scenario_result,
            parameter_changes=scenario_params,
            metrics_delta=metrics_delta
        )
        
    def _apply_modifications(
        self,
        missions: List[Dict],
        resources: Dict,
        constraints: Dict,
        params: Dict[str, Any]
    ) -> Tuple[List[Dict], Dict, Dict]:
        """
        Apply parameter modifications to the data.
        
        Args:
            missions: Mission data to modify
            resources: Resource data to modify
            constraints: Constraint data to modify
            params: Modification parameters
            
        Returns:
            Tuple of modified (missions, resources, constraints)
        """
        # Duration multiplier
        if 'task_duration_multiplier' in params:
            mult = params['task_duration_multiplier']
            for mission in missions:
                for task in mission['tasks']:
                    task['duration_hours'] *= mult
                    
        # Add random delays
        if 'random_delay_hours' in params:
            delay_range = params['random_delay_hours']
            random.seed(params.get('seed', 42))
            for mission in missions:
                for task in mission['tasks']:
                    delay = random.uniform(delay_range[0], delay_range[1])
                    task['duration_hours'] += delay
                    
        # Resource capacity changes
        if 'resource_capacity_changes' in params:
            changes = params['resource_capacity_changes']
            for res_type in ['facilities', 'crews', 'equipment']:
                if res_type in resources:
                    for res in resources[res_type]:
                        if res['id'] in changes:
                            res['capacity'] = changes[res['id']]
                            
        # Remove resources
        if 'remove_resources' in params:
            to_remove = params['remove_resources']
            for res_type in ['facilities', 'crews', 'equipment']:
                if res_type in resources:
                    resources[res_type] = [
                        r for r in resources[res_type] 
                        if r['id'] not in to_remove
                    ]
                    
        # Add resources
        if 'add_resources' in params:
            for res in params['add_resources']:
                res_type = res.get('type', 'equipment')
                if res_type in resources:
                    resources[res_type].append(res)
                    
        # Modify launch windows
        if 'launch_window_shifts' in params:
            shifts = params['launch_window_shifts']
            for mission in missions:
                if mission['id'] in shifts:
                    shift_days = shifts[mission['id']]
                    start = datetime.fromisoformat(mission['launch_window_start'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(mission['launch_window_end'].replace('Z', '+00:00'))
                    mission['launch_window_start'] = (start + timedelta(days=shift_days)).isoformat().replace('+00:00', 'Z')
                    mission['launch_window_end'] = (end + timedelta(days=shift_days)).isoformat().replace('+00:00', 'Z')
                    
        # Modify safety buffers
        if 'safety_buffer_multiplier' in params:
            mult = params['safety_buffer_multiplier']
            if 'safety_buffer_constraints' in constraints:
                if 'default_buffer_hours' in constraints['safety_buffer_constraints']:
                    for key in constraints['safety_buffer_constraints']['default_buffer_hours']:
                        constraints['safety_buffer_constraints']['default_buffer_hours'][key] *= mult
                        
        # Add maintenance windows
        if 'add_maintenance' in params:
            if 'maintenance_windows' not in constraints:
                constraints['maintenance_windows'] = []
            constraints['maintenance_windows'].extend(params['add_maintenance'])
            
        return missions, resources, constraints
        
    def _create_temp_scenario(
        self,
        missions: List[Dict],
        resources: Dict,
        constraints: Dict
    ) -> Path:
        """Create temporary directory with scenario data."""
        temp_dir = self.base_data_dir.parent / "temp_scenario"
        temp_dir.mkdir(exist_ok=True)
        
        with open(temp_dir / "missions.json", 'w') as f:
            json.dump({"missions": missions}, f, indent=2)
            
        with open(temp_dir / "resources.json", 'w') as f:
            json.dump(resources, f, indent=2)
            
        with open(temp_dir / "constraints.json", 'w') as f:
            json.dump(constraints, f, indent=2)
            
        return temp_dir
        
    def sensitivity_analysis(
        self,
        parameter_name: str,
        parameter_values: List[Any],
        modification_func: Callable = None
    ) -> SensitivityResult:
        """
        Perform sensitivity analysis on a parameter.
        
        Args:
            parameter_name: Name of parameter to vary
            parameter_values: List of values to test
            modification_func: Optional function to apply parameter value
            
        Returns:
            SensitivityResult with results for each parameter value
        """
        if self.base_result is None:
            self.load_base_case()
            
        makespans = []
        costs = []
        
        for value in parameter_values:
            # Create scenario with this parameter value
            if modification_func:
                params = modification_func(value)
            else:
                params = {parameter_name: value}
                
            try:
                result = self.run_scenario(params, f"sensitivity_{value}")
                makespans.append(result.scenario_result.makespan_hours)
                costs.append(result.scenario_result.total_cost)
            except Exception as e:
                print(f"Failed for {parameter_name}={value}: {e}")
                makespans.append(float('nan'))
                costs.append(float('nan'))
                
        return SensitivityResult(
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            makespan_values=makespans,
            cost_values=costs
        )
        
    def monte_carlo_simulation(
        self,
        uncertainty_params: Dict[str, Tuple[float, float]],
        n_simulations: int = 100,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulation with uncertain parameters.
        
        Args:
            uncertainty_params: Dict mapping param names to (mean, std) tuples
            n_simulations: Number of simulations to run
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with simulation results
        """
        random.seed(seed)
        np.random.seed(seed)
        
        results = []
        
        for i in range(n_simulations):
            # Sample uncertain parameters
            params = {}
            
            if 'duration_uncertainty' in uncertainty_params:
                mean, std = uncertainty_params['duration_uncertainty']
                params['task_duration_multiplier'] = max(0.5, np.random.normal(mean, std))
                
            if 'delay_uncertainty' in uncertainty_params:
                mean, std = uncertainty_params['delay_uncertainty']
                delay = max(0, np.random.normal(mean, std))
                params['random_delay_hours'] = [0, delay]
                params['seed'] = i
                
            try:
                result = self.run_scenario(params, f"mc_sim_{i}")
                results.append({
                    'simulation': i,
                    'makespan': result.scenario_result.makespan_hours,
                    'cost': result.scenario_result.total_cost,
                    'status': result.scenario_result.status,
                    **params
                })
            except Exception as e:
                results.append({
                    'simulation': i,
                    'makespan': float('nan'),
                    'cost': float('nan'),
                    'status': 'failed',
                    'error': str(e)
                })
                
        return pd.DataFrame(results)
        
    def resource_impact_analysis(
        self,
        resource_id: str,
        capacity_range: List[int]
    ) -> pd.DataFrame:
        """
        Analyze the impact of resource capacity changes.
        
        Args:
            resource_id: ID of resource to analyze
            capacity_range: List of capacity values to test
            
        Returns:
            DataFrame with results for each capacity level
        """
        results = []
        
        for capacity in capacity_range:
            params = {
                'resource_capacity_changes': {
                    resource_id: capacity
                }
            }
            
            try:
                result = self.run_scenario(
                    params, 
                    f"capacity_{resource_id}_{capacity}"
                )
                results.append({
                    'resource_id': resource_id,
                    'capacity': capacity,
                    'makespan_hours': result.scenario_result.makespan_hours,
                    'total_cost': result.scenario_result.total_cost,
                    'utilization': result.scenario_result.resource_utilization.get(resource_id, 0)
                })
            except Exception as e:
                results.append({
                    'resource_id': resource_id,
                    'capacity': capacity,
                    'makespan_hours': float('nan'),
                    'total_cost': float('nan'),
                    'utilization': float('nan'),
                    'error': str(e)
                })
                
        return pd.DataFrame(results)
        
    def delay_impact_analysis(
        self,
        task_id: str,
        delay_hours_range: List[float]
    ) -> pd.DataFrame:
        """
        Analyze the impact of task delays.
        
        Args:
            task_id: ID of task to delay
            delay_hours_range: List of delay durations to test
            
        Returns:
            DataFrame with results for each delay amount
        """
        results = []
        
        for delay in delay_hours_range:
            # Create modified missions with delayed task
            modified_missions = copy.deepcopy(self.base_missions)
            
            for mission in modified_missions:
                for task in mission['tasks']:
                    if task['id'] == task_id:
                        task['duration_hours'] += delay
                        
            temp_dir = self._create_temp_scenario(
                modified_missions,
                self.base_resources,
                self.base_constraints
            )
            
            try:
                scheduler = GroundOpsScheduler(data_dir=str(temp_dir))
                scheduler.load_data()
                scenario_result = scheduler.optimize()
                
                results.append({
                    'task_id': task_id,
                    'delay_hours': delay,
                    'makespan_hours': scenario_result.makespan_hours,
                    'total_cost': scenario_result.total_cost,
                    'makespan_increase': scenario_result.makespan_hours - self.base_result.makespan_hours
                })
            except Exception as e:
                results.append({
                    'task_id': task_id,
                    'delay_hours': delay,
                    'makespan_hours': float('nan'),
                    'total_cost': float('nan'),
                    'makespan_increase': float('nan'),
                    'error': str(e)
                })
                
        return pd.DataFrame(results)
        
    def compare_scenarios(
        self,
        scenarios: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple scenarios side by side.
        
        Args:
            scenarios: Dict mapping scenario names to parameter dicts
            
        Returns:
            DataFrame comparing all scenarios
        """
        results = []
        
        for name, params in scenarios.items():
            try:
                result = self.run_scenario(params, name)
                results.append({
                    'scenario': name,
                    'makespan_hours': result.scenario_result.makespan_hours,
                    'total_cost': result.scenario_result.total_cost,
                    'solve_time_seconds': result.scenario_result.solve_time_seconds,
                    'status': result.scenario_result.status,
                    'makespan_delta': result.metrics_delta['makespan_hours'],
                    'cost_delta': result.metrics_delta['total_cost']
                })
            except Exception as e:
                results.append({
                    'scenario': name,
                    'error': str(e)
                })
                
        return pd.DataFrame(results)
        
    def planned_vs_as_run(
        self,
        actual_durations: Dict[str, float],
        actual_start_times: Dict[str, datetime] = None
    ) -> Dict:
        """
        Compare planned schedule to actual (as-run) execution.
        
        This is directly relevant to the NASA job posting's
        "planned vs. as-run assessment" requirement.
        
        Args:
            actual_durations: Dict mapping task_id to actual duration
            actual_start_times: Optional dict of actual start times
            
        Returns:
            Dictionary with comparison metrics
        """
        if self.base_result is None:
            self.load_base_case()
            
        planned = self.base_result.tasks.copy()
        
        # Calculate actual schedule
        actual_schedule = []
        cumulative_delay = 0
        
        for _, row in planned.iterrows():
            task_id = row['task_id']
            planned_duration = row['duration_hours']
            actual_duration = actual_durations.get(task_id, planned_duration)
            duration_variance = actual_duration - planned_duration
            
            actual_schedule.append({
                'task_id': task_id,
                'task_name': row['task_name'],
                'planned_start': row['start_time'],
                'planned_duration': planned_duration,
                'actual_duration': actual_duration,
                'duration_variance': duration_variance,
                'variance_percent': (duration_variance / planned_duration * 100) if planned_duration > 0 else 0
            })
            
        df = pd.DataFrame(actual_schedule)
        
        return {
            'comparison_table': df,
            'total_duration_variance': df['duration_variance'].sum(),
            'average_variance_percent': df['variance_percent'].mean(),
            'max_delay_task': df.loc[df['duration_variance'].idxmax(), 'task_id'] if len(df) > 0 else None,
            'tasks_with_delays': len(df[df['duration_variance'] > 0])
        }


# Predefined scenario templates
def get_predefined_scenarios() -> Dict[str, Dict[str, Any]]:
    """Get predefined scenario templates for common what-if analyses."""
    return {
        "optimistic": {
            "task_duration_multiplier": 0.9,
            "safety_buffer_multiplier": 0.8
        },
        "pessimistic": {
            "task_duration_multiplier": 1.2,
            "safety_buffer_multiplier": 1.5
        },
        "reduced_crew": {
            "resource_capacity_changes": {
                "Assembly-Crew-A": 0,  # Crew unavailable
            }
        },
        "delayed_launch_window": {
            "launch_window_shifts": {
                "ARTEMIS-IV": 7  # 1 week delay
            }
        },
        "resource_shortage": {
            "task_duration_multiplier": 1.1,
            "safety_buffer_multiplier": 1.2
        },
        "expedited": {
            "task_duration_multiplier": 0.85,
            "safety_buffer_multiplier": 0.5
        }
    }
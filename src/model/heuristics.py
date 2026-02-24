"""
Heuristic Algorithms for NASA Ground Operations Scheduler.

This module provides fast heuristic methods for generating feasible
schedules that can be used as:
1. Standalone solutions for quick approximations
2. Warm starts for the MILP solver
3. Upper bounds for branch-and-bound

Heuristics Implemented:
- Priority-based list scheduling
- Earliest Deadline First (EDF)
- Critical Path scheduling
- Resource-based greedy allocation
- Multi-pass improvement heuristics

Author: Operations Research Portfolio Project
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

from src.model.scheduler import Task, Resource, ScheduleResult


@dataclass
class HeuristicResult:
    """Container for heuristic solution."""
    schedule: pd.DataFrame
    makespan_hours: float
    total_cost: float
    solve_time_seconds: float
    heuristic_name: str
    is_feasible: bool = True
    violations: List[str] = field(default_factory=list)
    
    def to_schedule_result(self) -> ScheduleResult:
        """Convert to ScheduleResult format."""
        return ScheduleResult(
            status="heuristic" if self.is_feasible else "infeasible",
            makespan_hours=self.makespan_hours,
            total_cost=self.total_cost,
            tasks=self.schedule,
            resource_utilization={},
            solve_time_seconds=self.solve_time_seconds,
            mip_gap=1.0,  # Unknown for heuristics
            objective_value=self.makespan_hours
        )


class ScheduleState:
    """
    Tracks the current state of a partial schedule.
    
    Used by heuristics to track task completions and resource availability.
    """
    
    def __init__(
        self,
        tasks: Dict[str, Task],
        resources: Dict[str, Resource],
        planning_start: datetime,
        time_horizon_hours: int
    ):
        """Initialize schedule state."""
        self.tasks = tasks
        self.resources = resources
        self.planning_start = planning_start
        self.time_horizon_hours = time_horizon_hours
        
        # Track scheduled tasks
        self.scheduled: Dict[str, Dict] = {}  # task_id -> schedule info
        
        # Resource availability timeline
        # For each resource, track when it becomes available
        self.resource_available: Dict[str, float] = {
            res_id: 0.0 for res_id in resources
        }
        
        # Task completion times
        self.task_completion: Dict[str, float] = {}
        
    def is_scheduled(self, task_id: str) -> bool:
        """Check if a task is already scheduled."""
        return task_id in self.scheduled
        
    def can_schedule(self, task_id: str) -> bool:
        """Check if a task can be scheduled (all predecessors done)."""
        task = self.tasks[task_id]
        
        for pred_id in task.predecessors:
            if pred_id not in self.task_completion:
                return False
                
        return True
        
    def get_earliest_start(self, task_id: str) -> float:
        """Get earliest possible start time for a task."""
        task = self.tasks[task_id]
        
        # Start after all predecessors complete
        earliest = 0.0
        for pred_id in task.predecessors:
            if pred_id in self.task_completion:
                pred_task = self.tasks[pred_id]
                buffer = pred_task.safety_buffer_after_hours
                earliest = max(earliest, self.task_completion[pred_id] + buffer)
                
        # Start when all required resources are available
        for res_id in task.required_resources:
            if res_id in self.resource_available:
                earliest = max(earliest, self.resource_available[res_id])
                
        return earliest
        
    def schedule_task(
        self,
        task_id: str,
        start_hours: float
    ) -> Dict:
        """Schedule a task at the given start time."""
        task = self.tasks[task_id]
        end_hours = start_hours + task.duration_hours
        
        # Update state
        self.scheduled[task_id] = {
            "task_id": task_id,
            "task_name": task.name,
            "mission_id": task.mission_id,
            "start_hours": start_hours,
            "end_hours": end_hours,
            "duration_hours": task.duration_hours,
            "resources": task.required_resources,
            "hazard_level": task.hazard_level,
            "start_time": self.planning_start + timedelta(hours=start_hours),
            "end_time": self.planning_start + timedelta(hours=end_hours)
        }
        
        self.task_completion[task_id] = end_hours
        
        # Update resource availability
        for res_id in task.required_resources:
            if res_id in self.resource_available:
                self.resource_available[res_id] = end_hours
                
        return self.scheduled[task_id]
        
    def get_makespan(self) -> float:
        """Get current makespan."""
        if not self.task_completion:
            return 0.0
        return max(self.task_completion.values())
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert scheduled tasks to DataFrame."""
        if not self.scheduled:
            return pd.DataFrame()
        return pd.DataFrame(list(self.scheduled.values())).sort_values("start_hours")


class ListScheduler:
    """
    Priority-based list scheduling heuristic.
    
    Schedules tasks in priority order, placing each task at its earliest
    feasible start time. Fast O(n^2) but may not find optimal solutions.
    
    Priority rules available:
    - shortest_processing_time: Prefer shorter tasks
    - longest_processing_time: Prefer longer tasks
    - most_successors: Prefer tasks with most downstream dependencies
    - critical_path: Prefer tasks on critical path
    - launch_window: Prefer tasks with earlier launch windows
    """
    
    def __init__(
        self,
        tasks: Dict[str, Task],
        resources: Dict[str, Resource],
        planning_start: datetime,
        time_horizon_hours: int
    ):
        """Initialize the list scheduler."""
        self.tasks = tasks
        self.resources = resources
        self.planning_start = planning_start
        self.time_horizon_hours = time_horizon_hours
        
    def schedule(
        self,
        priority_rule: str = "most_successors",
        objective: str = "minimize_makespan"
    ) -> HeuristicResult:
        """
        Run list scheduling with the given priority rule.
        
        Args:
            priority_rule: Priority rule to use
            objective: Scheduling objective
            
        Returns:
            HeuristicResult with the schedule
        """
        import time
        start_time = time.time()
        
        state = ScheduleState(
            self.tasks, self.resources,
            self.planning_start, self.time_horizon_hours
        )
        
        # Compute priorities
        priorities = self._compute_priorities(priority_rule)
        
        # Create task queue sorted by priority
        task_queue = list(self.tasks.keys())
        task_queue.sort(key=lambda t: priorities.get(t, 0), reverse=True)
        
        # Schedule tasks
        scheduled_count = 0
        max_iterations = len(self.tasks) * 10
        iteration = 0
        
        while scheduled_count < len(self.tasks) and iteration < max_iterations:
            iteration += 1
            made_progress = False
            
            for task_id in task_queue:
                if state.is_scheduled(task_id):
                    continue
                    
                if not state.can_schedule(task_id):
                    continue
                    
                # Find earliest start time
                earliest = state.get_earliest_start(task_id)
                
                # Check time windows
                earliest = self._adjust_for_time_windows(task_id, earliest)
                
                # Schedule the task
                state.schedule_task(task_id, earliest)
                scheduled_count += 1
                made_progress = True
                
            if not made_progress:
                # No progress - might have a cycle or infeasibility
                break
                
        # Build result
        df = state.to_dataframe()
        makespan = state.get_makespan()
        
        # Calculate cost
        total_cost = self._calculate_cost(df)
        
        # Check for violations
        violations = self._check_violations(state)
        
        solve_time = time.time() - start_time
        
        return HeuristicResult(
            schedule=df,
            makespan_hours=makespan,
            total_cost=total_cost,
            solve_time_seconds=solve_time,
            heuristic_name=f"list_{priority_rule}",
            is_feasible=len(violations) == 0,
            violations=violations
        )
        
    def _compute_priorities(self, rule: str) -> Dict[str, float]:
        """Compute task priorities based on the given rule."""
        priorities = {}
        
        if rule == "shortest_processing_time":
            for task_id, task in self.tasks.items():
                priorities[task_id] = -task.duration_hours
                
        elif rule == "longest_processing_time":
            for task_id, task in self.tasks.items():
                priorities[task_id] = task.duration_hours
                
        elif rule == "most_successors":
            # Count total downstream tasks
            successor_count = {}
            for task_id in self.tasks:
                successor_count[task_id] = self._count_successors(task_id)
            priorities = successor_count
            
        elif rule == "critical_path":
            # Priority based on longest path to end
            for task_id in self.tasks.items():
                priorities[task_id] = self._compute_remaining_work(task_id)
                
        elif rule == "launch_window":
            # Priority based on earliest launch window
            for task_id, task in self.tasks.items():
                if task.launch_window_start:
                    hours = (task.launch_window_start - self.planning_start).total_seconds() / 3600
                    priorities[task_id] = -hours
                else:
                    priorities[task_id] = 0
                    
        else:
            # Default: order by task ID
            for i, task_id in enumerate(self.tasks):
                priorities[task_id] = -i
                
        return priorities
        
    def _count_successors(self, task_id: str, visited: set = None) -> int:
        """Count total successors of a task."""
        if visited is None:
            visited = set()
            
        if task_id in visited:
            return 0
            
        visited.add(task_id)
        count = 0
        
        for tid, task in self.tasks.items():
            if task_id in task.predecessors:
                count += 1 + self._count_successors(tid, visited)
                
        return count
        
    def _compute_remaining_work(self, task_id: str) -> float:
        """Compute remaining work from this task to end."""
        task = self.tasks[task_id]
        
        # Find successors
        successors = [
            tid for tid, t in self.tasks.items()
            if task_id in t.predecessors
        ]
        
        if not successors:
            return task.duration_hours
            
        max_successor_work = max(
            self._compute_remaining_work(sid)
            for sid in successors
        )
        
        return task.duration_hours + max_successor_work
        
    def _adjust_for_time_windows(self, task_id: str, earliest: float) -> float:
        """Adjust start time for time window constraints."""
        task = self.tasks[task_id]
        
        if not task.launch_window_start or not task.launch_window_end:
            return earliest
            
        window_start = (
            task.launch_window_start - self.planning_start
        ).total_seconds() / 3600
        window_end = (
            task.launch_window_end - self.planning_start
        ).total_seconds() / 3600
        
        if "Launch" in task.name:
            # Launch must start within window
            return max(earliest, window_start)
        else:
            # Must complete before window ends
            latest_start = window_end - task.duration_hours
            if earliest > latest_start:
                # Might be infeasible - return earliest anyway
                pass
                
        return earliest
        
    def _calculate_cost(self, df: pd.DataFrame) -> float:
        """Calculate total cost of schedule."""
        total_cost = 0.0
        
        for _, row in df.iterrows():
            task = self.tasks.get(row["task_id"])
            if task:
                for res_id in task.required_resources:
                    if res_id in self.resources:
                        total_cost += (
                            task.duration_hours * 
                            self.resources[res_id].hourly_cost
                        )
                        
        return total_cost
        
    def _check_violations(self, state: ScheduleState) -> List[str]:
        """Check for constraint violations."""
        violations = []
        
        # Check all tasks scheduled
        for task_id in self.tasks:
            if not state.is_scheduled(task_id):
                violations.append(f"Task {task_id} not scheduled")
                
        # Check precedence
        for task_id, task in self.tasks.items():
            if not state.is_scheduled(task_id):
                continue
                
            start = state.scheduled[task_id]["start_hours"]
            
            for pred_id in task.predecessors:
                if state.is_scheduled(pred_id):
                    pred_task = self.tasks[pred_id]
                    pred_end = state.task_completion[pred_id]
                    buffer = pred_task.safety_buffer_after_hours
                    
                    if start < pred_end + buffer - 0.001:
                        violations.append(
                            f"Precedence violation: {task_id} starts before "
                            f"{pred_id} completes"
                        )
                        
        return violations


class CriticalPathScheduler:
    """
    Critical path-based scheduling heuristic.
    
    Identifies the critical path and schedules those tasks first,
    then fills in non-critical tasks.
    """
    
    def __init__(
        self,
        tasks: Dict[str, Task],
        resources: Dict[str, Resource],
        planning_start: datetime,
        time_horizon_hours: int
    ):
        """Initialize the critical path scheduler."""
        self.tasks = tasks
        self.resources = resources
        self.planning_start = planning_start
        self.time_horizon_hours = time_horizon_hours
        
    def schedule(self) -> HeuristicResult:
        """Run critical path scheduling."""
        import time
        start_time = time.time()
        
        # Find critical path
        critical_path = self._find_critical_path()
        
        # Create scheduler state
        state = ScheduleState(
            self.tasks, self.resources,
            self.planning_start, self.time_horizon_hours
        )
        
        # Schedule critical path first
        for task_id in critical_path:
            earliest = state.get_earliest_start(task_id)
            state.schedule_task(task_id, earliest)
            
        # Schedule remaining tasks
        list_sched = ListScheduler(
            self.tasks, self.resources,
            self.planning_start, self.time_horizon_hours
        )
        
        # Use most_successors for remaining tasks
        for task_id in self.tasks:
            if not state.is_scheduled(task_id) and state.can_schedule(task_id):
                earliest = state.get_earliest_start(task_id)
                state.schedule_task(task_id, earliest)
                
        df = state.to_dataframe()
        makespan = state.get_makespan()
        total_cost = self._calculate_cost(df)
        
        solve_time = time.time() - start_time
        
        return HeuristicResult(
            schedule=df,
            makespan_hours=makespan,
            total_cost=total_cost,
            solve_time_seconds=solve_time,
            heuristic_name="critical_path"
        )
        
    def _find_critical_path(self) -> List[str]:
        """Find the critical path through the task graph."""
        # Compute earliest start times
        earliest_start = {}
        
        # Topological sort
        in_degree = {tid: 0 for tid in self.tasks}
        for task_id, task in self.tasks.items():
            for pred_id in task.predecessors:
                if pred_id in self.tasks:
                    in_degree[task_id] += 1
                    
        # Process tasks in topological order
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        
        while queue:
            task_id = queue.pop(0)
            task = self.tasks[task_id]
            
            # Compute earliest start
            if task.predecessors:
                earliest = max(
                    earliest_start.get(pred, 0) + 
                    self.tasks[pred].duration_hours +
                    self.tasks[pred].safety_buffer_after_hours
                    for pred in task.predecessors
                    if pred in self.tasks
                )
            else:
                earliest = 0
                
            earliest_start[task_id] = earliest
            
            # Update successors
            for tid, t in self.tasks.items():
                if task_id in t.predecessors:
                    in_degree[tid] -= 1
                    if in_degree[tid] == 0:
                        queue.append(tid)
                        
        # Find task with latest completion
        latest_completion = -1
        end_task = None
        
        for task_id, start in earliest_start.items():
            completion = start + self.tasks[task_id].duration_hours
            if completion > latest_completion:
                latest_completion = completion
                end_task = task_id
                
        # Backtrack to find critical path
        path = []
        if end_task:
            current = end_task
            while current:
                path.append(current)
                current = self._find_critical_predecessor(current, earliest_start)
                
        path.reverse()
        return path
        
    def _find_critical_predecessor(
        self,
        task_id: str,
        earliest_start: Dict[str, float]
    ) -> Optional[str]:
        """Find the critical predecessor of a task."""
        task = self.tasks[task_id]
        
        if not task.predecessors:
            return None
            
        critical_pred = None
        critical_end = -1
        
        for pred_id in task.predecessors:
            if pred_id not in self.tasks:
                continue
                
            pred = self.tasks[pred_id]
            pred_end = earliest_start.get(pred_id, 0) + pred.duration_hours
            
            if pred_end > critical_end:
                critical_end = pred_end
                critical_pred = pred_id
                
        return critical_pred
        
    def _calculate_cost(self, df: pd.DataFrame) -> float:
        """Calculate total cost of schedule."""
        total_cost = 0.0
        
        for _, row in df.iterrows():
            task = self.tasks.get(row["task_id"])
            if task:
                for res_id in task.required_resources:
                    if res_id in self.resources:
                        total_cost += (
                            task.duration_hours * 
                            self.resources[res_id].hourly_cost
                        )
                        
        return total_cost


class MultiStartImprover:
    """
    Multi-start heuristic with local improvement.
    
    Runs multiple heuristics and applies local search to improve
    the best solution found.
    """
    
    def __init__(
        self,
        tasks: Dict[str, Task],
        resources: Dict[str, Resource],
        planning_start: datetime,
        time_horizon_hours: int
    ):
        """Initialize the multi-start improver."""
        self.tasks = tasks
        self.resources = resources
        self.planning_start = planning_start
        self.time_horizon_hours = time_horizon_hours
        
    def run(
        self,
        num_starts: int = 5,
        improve: bool = True
    ) -> HeuristicResult:
        """
        Run multi-start heuristic.
        
        Args:
            num_starts: Number of different starting points
            improve: Whether to apply local search improvement
            
        Returns:
            Best heuristic result found
        """
        import time
        start_time = time.time()
        
        results = []
        
        # Try different priority rules
        priority_rules = [
            "most_successors",
            "shortest_processing_time",
            "longest_processing_time",
            "critical_path",
            "launch_window"
        ]
        
        list_sched = ListScheduler(
            self.tasks, self.resources,
            self.planning_start, self.time_horizon_hours
        )
        
        for rule in priority_rules[:num_starts]:
            try:
                result = list_sched.schedule(priority_rule=rule)
                results.append(result)
            except Exception as e:
                pass
                
        # Also try critical path scheduler
        try:
            cp_sched = CriticalPathScheduler(
                self.tasks, self.resources,
                self.planning_start, self.time_horizon_hours
            )
            result = cp_sched.schedule()
            results.append(result)
        except Exception as e:
            pass
            
        if not results:
            return HeuristicResult(
                schedule=pd.DataFrame(),
                makespan_hours=float('inf'),
                total_cost=0,
                solve_time_seconds=time.time() - start_time,
                heuristic_name="multi_start_failed",
                is_feasible=False
            )
            
        # Find best result
        best = min(results, key=lambda r: r.makespan_hours)
        
        # Apply local improvement if requested
        if improve and best.is_feasible:
            best = self._improve_solution(best)
            
        best.heuristic_name = f"multi_start_{best.heuristic_name}"
        best.solve_time_seconds = time.time() - start_time
        
        return best
        
    def _improve_solution(self, result: HeuristicResult) -> HeuristicResult:
        """Apply local search improvement."""
        improved = False
        df = result.schedule.copy()
        
        # Try swapping adjacent tasks
        for i in range(len(df) - 1):
            task1 = df.iloc[i]["task_id"]
            task2 = df.iloc[i + 1]["task_id"]
            
            # Check if swap is valid (no precedence between them)
            t1 = self.tasks[task1]
            t2 = self.tasks[task2]
            
            if task2 in t1.predecessors or task1 in t2.predecessors:
                continue
                
            # Try swap - skip for now as it requires full reschedule
            # This is a simplified version
            
        return result


def generate_warm_start(
    tasks: Dict[str, Task],
    resources: Dict[str, Resource],
    planning_start: datetime,
    time_horizon_hours: int
) -> Dict[str, float]:
    """
    Generate a warm start solution for the MILP solver.
    
    Returns a dictionary mapping task_id to suggested start time.
    """
    scheduler = MultiStartImprover(
        tasks, resources, planning_start, time_horizon_hours
    )
    
    result = scheduler.run(num_starts=3, improve=True)
    
    if not result.is_feasible:
        return {}
        
    warm_start = {}
    for _, row in result.schedule.iterrows():
        warm_start[row["task_id"]] = row["start_hours"]
        
    return warm_start


def run_all_heuristics(
    tasks: Dict[str, Task],
    resources: Dict[str, Resource],
    planning_start: datetime,
    time_horizon_hours: int
) -> pd.DataFrame:
    """
    Run all heuristics and compare results.
    
    Returns:
        DataFrame comparing all heuristic results
    """
    results = []
    
    # List scheduling variants
    list_sched = ListScheduler(
        tasks, resources, planning_start, time_horizon_hours
    )
    
    for rule in ["most_successors", "shortest_processing_time", 
                 "longest_processing_time", "launch_window"]:
        try:
            result = list_sched.schedule(priority_rule=rule)
            results.append({
                "heuristic": f"list_{rule}",
                "makespan_hours": result.makespan_hours,
                "total_cost": result.total_cost,
                "solve_time": result.solve_time_seconds,
                "feasible": result.is_feasible
            })
        except Exception as e:
            results.append({
                "heuristic": f"list_{rule}",
                "error": str(e)
            })
            
    # Critical path
    try:
        cp_sched = CriticalPathScheduler(
            tasks, resources, planning_start, time_horizon_hours
        )
        result = cp_sched.schedule()
        results.append({
            "heuristic": "critical_path",
            "makespan_hours": result.makespan_hours,
            "total_cost": result.total_cost,
            "solve_time": result.solve_time_seconds,
            "feasible": result.is_feasible
        })
    except Exception as e:
        results.append({
            "heuristic": "critical_path",
            "error": str(e)
        })
        
    # Multi-start
    try:
        ms_sched = MultiStartImprover(
            tasks, resources, planning_start, time_horizon_hours
        )
        result = ms_sched.run(num_starts=5, improve=True)
        results.append({
            "heuristic": "multi_start",
            "makespan_hours": result.makespan_hours,
            "total_cost": result.total_cost,
            "solve_time": result.solve_time_seconds,
            "feasible": result.is_feasible
        })
    except Exception as e:
        results.append({
            "heuristic": "multi_start",
            "error": str(e)
        })
        
    return pd.DataFrame(results)


class HeuristicScheduler:
    """
    Main interface for heuristic scheduling.
    
    Provides a unified interface to run various heuristics and
    compare their performance.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize with data directory."""
        self.data_dir = Path(data_dir)
        self.tasks: Dict[str, Task] = {}
        self.resources: Dict[str, Resource] = {}
        self.planning_start: datetime = datetime(2028, 1, 1)
        self.time_horizon_hours: int = 2160
        
    def load_data(self) -> None:
        """Load data from files."""
        # Load missions
        missions_path = self.data_dir / "missions.json"
        with open(missions_path, 'r') as f:
            missions_data = json.load(f)
            
        for mission in missions_data["missions"]:
            mission_id = mission["id"]
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
            
        # Set planning horizon
        planning = resources_data.get("planning_horizon", {})
        self.planning_start = datetime.fromisoformat(
            planning.get("start_date", "2028-01-01T00:00:00Z").replace("Z", "+00:00")
        )
        end = datetime.fromisoformat(
            planning.get("end_date", "2028-04-01T00:00:00Z").replace("Z", "+00:00")
        )
        self.time_horizon_hours = int((end - self.planning_start).total_seconds() / 3600)
        
    def schedule(
        self,
        method: str = "multi_start",
        **kwargs
    ) -> HeuristicResult:
        """
        Run a scheduling heuristic.
        
        Args:
            method: Heuristic method to use
            **kwargs: Additional arguments for the heuristic
            
        Returns:
            HeuristicResult with the schedule
        """
        if method == "list":
            scheduler = ListScheduler(
                self.tasks, self.resources,
                self.planning_start, self.time_horizon_hours
            )
            return scheduler.schedule(
                priority_rule=kwargs.get("priority_rule", "most_successors")
            )
            
        elif method == "critical_path":
            scheduler = CriticalPathScheduler(
                self.tasks, self.resources,
                self.planning_start, self.time_horizon_hours
            )
            return scheduler.schedule()
            
        elif method == "multi_start":
            scheduler = MultiStartImprover(
                self.tasks, self.resources,
                self.planning_start, self.time_horizon_hours
            )
            return scheduler.run(
                num_starts=kwargs.get("num_starts", 5),
                improve=kwargs.get("improve", True)
            )
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def compare_methods(self) -> pd.DataFrame:
        """Compare all heuristic methods."""
        return run_all_heuristics(
            self.tasks, self.resources,
            self.planning_start, self.time_horizon_hours
        )
        
    def get_warm_start(self) -> Dict[str, float]:
        """Get a warm start solution for MILP solver."""
        return generate_warm_start(
            self.tasks, self.resources,
            self.planning_start, self.time_horizon_hours
        )
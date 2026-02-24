"""Test script for the NASA Ground Operations Optimizer improvements."""

import sys
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Add project root to path
sys.path.insert(0, str(SCRIPT_DIR))

# Data directory (relative to script location)
DATA_DIR = SCRIPT_DIR / "data"

def test_heuristics():
    """Test the heuristic scheduler."""
    print("=" * 60)
    print("Testing Heuristic Scheduler")
    print("=" * 60)
    
    from src.model.heuristics import HeuristicScheduler
    
    scheduler = HeuristicScheduler(data_dir=str(DATA_DIR))
    scheduler.load_data()
    
    print(f"\nTasks loaded: {len(scheduler.tasks)}")
    print(f"Resources loaded: {len(scheduler.resources)}")
    print(f"Planning horizon: {scheduler.time_horizon_hours} hours")
    
    # Run list scheduling
    print("\n--- List Scheduling (most_successors) ---")
    result = scheduler.schedule(method="list", priority_rule="most_successors")
    
    print(f"Heuristic: {result.heuristic_name}")
    print(f"Makespan: {result.makespan_hours:.1f} hours")
    print(f"Feasible: {result.is_feasible}")
    print(f"Solve time: {result.solve_time_seconds:.4f} seconds")
    
    if len(result.schedule) > 0:
        print("\nSchedule preview (first 10 tasks):")
        cols = ['task_id', 'task_name', 'start_hours', 'duration_hours']
        print(result.schedule[cols].head(10).to_string())
        
    return result


def test_critical_path():
    """Test critical path scheduler."""
    print("\n" + "=" * 60)
    print("Testing Critical Path Scheduler")
    print("=" * 60)
    
    from src.model.heuristics import HeuristicScheduler
    
    scheduler = HeuristicScheduler(data_dir=str(DATA_DIR))
    scheduler.load_data()
    
    result = scheduler.schedule(method="critical_path")
    
    print(f"Heuristic: {result.heuristic_name}")
    print(f"Makespan: {result.makespan_hours:.1f} hours")
    print(f"Feasible: {result.is_feasible}")
    print(f"Solve time: {result.solve_time_seconds:.4f} seconds")
    
    return result


def test_multi_start():
    """Test multi-start heuristic."""
    print("\n" + "=" * 60)
    print("Testing Multi-Start Heuristic")
    print("=" * 60)
    
    from src.model.heuristics import HeuristicScheduler
    
    scheduler = HeuristicScheduler(data_dir=str(DATA_DIR))
    scheduler.load_data()
    
    result = scheduler.schedule(method="multi_start", num_starts=3, improve=False)
    
    print(f"Heuristic: {result.heuristic_name}")
    print(f"Makespan: {result.makespan_hours:.1f} hours")
    print(f"Feasible: {result.is_feasible}")
    print(f"Solve time: {result.solve_time_seconds:.4f} seconds")
    
    return result


def test_compare_heuristics():
    """Compare all heuristic methods."""
    print("\n" + "=" * 60)
    print("Comparing All Heuristic Methods")
    print("=" * 60)
    
    from src.model.heuristics import HeuristicScheduler
    
    scheduler = HeuristicScheduler(data_dir=str(DATA_DIR))
    scheduler.load_data()
    
    df = scheduler.compare_methods()
    print("\n" + df.to_string())
    
    return df


def test_performance_monitor():
    """Test performance monitoring."""
    print("\n" + "=" * 60)
    print("Testing Performance Monitor")
    print("=" * 60)
    
    from src.model.performance import PerformanceMonitor, PerformanceReport
    from src.model.heuristics import HeuristicScheduler
    from datetime import datetime
    import time
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    # Run a heuristic as the "optimization"
    scheduler = HeuristicScheduler(data_dir=str(DATA_DIR))
    scheduler.load_data()
    result = scheduler.schedule(method="list")
    
    monitor.stop()
    
    print(f"\nElapsed time: {monitor.get_elapsed_time():.4f} seconds")
    mem_stats = monitor.get_memory_stats()
    print(f"Peak memory: {mem_stats.peak_memory_mb:.1f} MB")
    print(f"Avg memory: {mem_stats.avg_memory_mb:.1f} MB")
    
    return monitor


def test_schedule_state():
    """Test the schedule state tracking."""
    print("\n" + "=" * 60)
    print("Testing Schedule State")
    print("=" * 60)
    
    from src.model.heuristics import HeuristicScheduler, ScheduleState
    from datetime import datetime
    
    scheduler = HeuristicScheduler(data_dir=str(DATA_DIR))
    scheduler.load_data()
    
    state = ScheduleState(
        scheduler.tasks,
        scheduler.resources,
        scheduler.planning_start,
        scheduler.time_horizon_hours
    )
    
    # Schedule first task
    first_task = list(scheduler.tasks.keys())[0]
    state.schedule_task(first_task, 0)
    
    print(f"\nScheduled task: {first_task}")
    print(f"Makespan: {state.get_makespan()} hours")
    print(f"Tasks scheduled: {len(state.scheduled)}")
    
    return state


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# NASA Ground Operations Optimizer - Improvement Tests")
    print("#" * 60)
    
    try:
        # Test schedule state
        test_schedule_state()
        
        # Test heuristics
        test_heuristics()
        test_critical_path()
        test_multi_start()
        
        # Compare all methods
        test_compare_heuristics()
        
        # Test performance monitoring
        test_performance_monitor()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
Unit tests for NASA Ground Operations Scheduler.

Run with: pytest tests/ -v

Author: Operations Research Portfolio Project
"""

import pytest
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.scheduler import GroundOpsScheduler, Task, Resource, ScheduleResult
from src.model.constraints import ConstraintBuilder, validate_constraint_config, get_default_constraint_config
from src.model.objectives import ObjectiveBuilder, ObjectiveType, calculate_schedule_metrics


class TestTask:
    """Tests for Task dataclass."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            id="TEST-01",
            name="Test Task",
            mission_id="TEST-MISSION",
            duration_hours=24,
            required_resources=["Resource-A"]
        )
        
        assert task.id == "TEST-01"
        assert task.name == "Test Task"
        assert task.duration_hours == 24
        assert task.hazard_level == "low"
        assert task.predecessors == []
        
    def test_task_hash(self):
        """Test task is hashable."""
        task1 = Task(
            id="TEST-01",
            name="Task 1",
            mission_id="M1",
            duration_hours=10,
            required_resources=[]
        )
        task2 = Task(
            id="TEST-02",
            name="Task 2",
            mission_id="M1",
            duration_hours=10,
            required_resources=[]
        )
        
        task_set = {task1, task2}
        assert len(task_set) == 2


class TestResource:
    """Tests for Resource dataclass."""
    
    def test_resource_creation(self):
        """Test basic resource creation."""
        resource = Resource(
            id="RES-01",
            name="Test Resource",
            type="facility",
            capacity=2,
            hourly_cost=1000
        )
        
        assert resource.id == "RES-01"
        assert resource.capacity == 2
        assert resource.hourly_cost == 1000


class TestGroundOpsScheduler:
    """Tests for GroundOpsScheduler class."""
    
    @pytest.fixture
    def scheduler(self):
        """Create scheduler instance with test data."""
        data_dir = Path(__file__).parent.parent / "data"
        scheduler = GroundOpsScheduler(data_dir=str(data_dir))
        return scheduler
    
    def test_load_data(self, scheduler):
        """Test data loading."""
        scheduler.load_data()
        
        assert len(scheduler.tasks) > 0
        assert len(scheduler.resources) > 0
        assert len(scheduler.missions) > 0
        
    def test_mission_tasks_loaded(self, scheduler):
        """Test that mission tasks are correctly loaded."""
        scheduler.load_data()
        
        # Check Artemis IV tasks
        artemis_tasks = [t for t in scheduler.tasks.values() if t.mission_id == "ARTEMIS-IV"]
        assert len(artemis_tasks) == 13  # 13 tasks in the mission
        
    def test_predecessors_loaded(self, scheduler):
        """Test that predecessors are correctly loaded."""
        scheduler.load_data()
        
        # Find a task with predecessors
        stacking_task = scheduler.tasks.get("ARTEMIS-IV-02")
        assert stacking_task is not None
        assert "ARTEMIS-IV-01" in stacking_task.predecessors
        
    def test_resources_by_type(self, scheduler):
        """Test that resources of different types are loaded."""
        scheduler.load_data()
        
        facility_count = sum(1 for r in scheduler.resources.values() if r.type == "facility")
        crew_count = sum(1 for r in scheduler.resources.values() if r.type == "crew")
        equipment_count = sum(1 for r in scheduler.resources.values() if r.type == "equipment")
        
        assert facility_count > 0
        assert crew_count > 0
        assert equipment_count > 0
        
    def test_planning_horizon(self, scheduler):
        """Test planning horizon is set correctly."""
        scheduler.load_data()
        
        assert scheduler.time_horizon_hours > 0
        assert scheduler.planning_start < scheduler.planning_end


class TestConstraintBuilder:
    """Tests for ConstraintBuilder class."""
    
    def test_default_config(self):
        """Test default constraint configuration."""
        config = get_default_constraint_config()
        
        assert "precedence_constraints" in config
        assert "resource_capacity_constraints" in config
        assert config["precedence_constraints"]["enforced"] == True
        
    def test_validate_config(self):
        """Test config validation."""
        valid_config = get_default_constraint_config()
        warnings = validate_constraint_config(valid_config)
        assert len(warnings) == 0
        
        # Missing required section
        invalid_config = {"precedence_constraints": {"enforced": True}}
        warnings = validate_constraint_config(invalid_config)
        assert len(warnings) > 0


class TestObjectiveBuilder:
    """Tests for ObjectiveBuilder class."""
    
    def test_objective_types(self):
        """Test objective type enum."""
        assert ObjectiveType.MINIMIZE_MAKESPAN.value == "minimize_makespan"
        assert ObjectiveType.MINIMIZE_COST.value == "minimize_cost"


class TestScheduleMetrics:
    """Tests for schedule metrics calculation."""
    
    def test_empty_schedule(self):
        """Test metrics with empty schedule."""
        import pandas as pd
        
        df = pd.DataFrame()
        metrics = calculate_schedule_metrics(df, {}, datetime.now())
        
        assert len(metrics) == 0


class TestJSONData:
    """Tests for JSON data files."""
    
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent.parent / "data"
    
    def test_missions_json_valid(self, data_dir):
        """Test missions.json is valid JSON."""
        with open(data_dir / "missions.json", 'r') as f:
            data = json.load(f)
            
        assert "missions" in data
        assert len(data["missions"]) > 0
        
    def test_resources_json_valid(self, data_dir):
        """Test resources.json is valid JSON."""
        with open(data_dir / "resources.json", 'r') as f:
            data = json.load(f)
            
        assert "facilities" in data
        assert "crews" in data
        assert "equipment" in data
        
    def test_constraints_json_valid(self, data_dir):
        """Test constraints.json is valid JSON."""
        with open(data_dir / "constraints.json", 'r') as f:
            data = json.load(f)
            
        assert "precedence_constraints" in data
        assert "optimization_settings" in data
        
    def test_mission_structure(self, data_dir):
        """Test mission data structure."""
        with open(data_dir / "missions.json", 'r') as f:
            data = json.load(f)
            
        for mission in data["missions"]:
            assert "id" in mission
            assert "name" in mission
            assert "tasks" in mission
            assert "launch_window_start" in mission
            assert "launch_window_end" in mission
            
            for task in mission["tasks"]:
                assert "id" in task
                assert "name" in task
                assert "duration_hours" in task
                assert "required_resources" in task


# Integration test (requires Gurobi)
@pytest.mark.skipif(
    not pytest.importorskip("gurobipy"),
    reason="Gurobi not available"
)
class TestOptimization:
    """Integration tests requiring Gurobi."""
    
    @pytest.fixture
    def scheduler(self):
        data_dir = Path(__file__).parent.parent / "data"
        scheduler = GroundOpsScheduler(data_dir=str(data_dir))
        scheduler.load_data()
        return scheduler
    
    def test_model_builds(self, scheduler):
        """Test that model builds without errors."""
        scheduler.build_model(objective="minimize_makespan")
        assert scheduler.model is not None
        
    def test_model_has_variables(self, scheduler):
        """Test that model has decision variables."""
        scheduler.build_model()
        
        assert len(scheduler.start_time_vars) == len(scheduler.tasks)
        assert scheduler.completion_time_var is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# 🚀 NASA Ground Operations Optimizer

A Mixed-Integer Linear Programming (MILP) optimization tool for scheduling space mission ground operations using Gurobi. This project demonstrates operations research techniques applicable to NASA's Exploration Ground Systems (EGS) program.

## 📋 Problem Description

Ground operations for space missions involve complex scheduling of tasks across limited resources:
- **Facilities**: Vehicle Assembly Building (VAB), launch pads, processing hangars
- **Crews**: Specialized teams for assembly, testing, fueling, launch operations
- **Equipment**: Cranes, transporters, ground support equipment

The optimizer schedules mission processing activities while respecting:
- Precedence constraints (assembly → testing → fueling → launch)
- Resource capacity limits
- Launch windows and crew shift schedules
- Safety buffer requirements between hazardous operations

## 🎯 Features

- **MILP Optimization**: Gurobi-based scheduler with multiple objective functions
- **Resource Allocation**: Optimal assignment of crews, facilities, and equipment
- **What-If Analysis**: Scenario simulation for schedule robustness
- **Interactive Dashboard**: Streamlit-based visualization with Gantt charts and metrics
- **Sensitivity Analysis**: Understand how parameter changes affect the schedule

## 🏗️ Project Structure

```
nasa-ground-ops-optimizer/
├── data/                      # Input data files
│   ├── missions.json          # Mission definitions
│   ├── resources.json         # Available resources
│   └── constraints.json       # Scheduling constraints
├── src/
│   ├── model/                 # Optimization models
│   │   ├── scheduler.py       # Main MILP model
│   │   ├── constraints.py     # Constraint definitions
│   │   └── objectives.py      # Objective functions
│   ├── simulation/            # What-if simulation
│   ├── analysis/              # Metrics and analysis
│   └── utils/                 # Utilities
├── dashboard/                 # Streamlit dashboard
├── notebooks/                 # Jupyter notebooks
├── tests/                     # Unit tests
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Gurobi Optimizer (academic license available)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd nasa-ground-ops-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Optimizer

```bash
# Run optimization
python -m src.model.scheduler

# Launch dashboard
streamlit run dashboard/app.py
```

## 📊 Model Formulation

### Decision Variables

- $s_{t}$: Start time of task $t$
- $x_{t,r}$: Binary variable = 1 if task $t$ uses resource $r$
- $y_{t_1,t_2}$: Binary variable = 1 if task $t_1$ precedes $t_2$ (for disjunctions)

### Constraints

1. **Precedence**: $s_{t_2} \geq s_{t_1} + d_{t_1}$ for all $(t_1, t_2) \in$ precedence pairs
2. **Resource Capacity**: $\sum_{t: s_t \leq \tau < s_t + d_t} x_{t,r} \leq C_r$ for each resource $r$
3. **Time Windows**: $TW^{min}_t \leq s_t \leq TW^{max}_t$
4. **Safety Buffers**: $s_{t_2} \geq s_{t_1} + d_{t_1} + buffer$ for hazardous sequences

### Objectives

- **Minimize Makespan**: $\min \max_t (s_t + d_t)$
- **Maximize Utilization**: $\max \sum_{t,r} x_{t,r} \cdot d_t$
- **Minimize Cost**: $\min \sum_{t,r} c_r \cdot x_{t,r} \cdot d_t$

## 📈 Results Visualization

The dashboard provides:
- Gantt chart of scheduled tasks
- Resource utilization heatmap
- Critical path analysis
- Scenario comparison tools

## 🛠️ Technologies Used

- **Optimization**: Gurobi, PuLP
- **Simulation**: SimPy
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Analysis**: Pandas, NumPy

## 📝 License

MIT License

## 👤 Author

Randy - Operations Research Portfolio Project

"""
NASA Ground Operations Optimizer - Interactive Dashboard

This Streamlit dashboard provides visualization and analysis tools for
the ground operations scheduling optimization system.

Features:
- Interactive Gantt chart visualization
- Resource utilization analysis
- What-if scenario comparison
- Sensitivity analysis tools
- Planned vs. as-run assessment

Run with: streamlit run dashboard/app.py

Author: Operations Research Portfolio Project
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.scheduler import GroundOpsScheduler, ScheduleResult
from src.simulation.what_if import WhatIfAnalyzer, get_predefined_scenarios


# Page config
st.set_page_config(
    page_title="NASA Ground Ops Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0B3D91;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0B3D91;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_scheduler(data_dir: str = "data"):
    """Load and cache the scheduler instance."""
    scheduler = GroundOpsScheduler(data_dir=data_dir)
    scheduler.load_data()
    return scheduler


@st.cache_data
def run_optimization(_scheduler, objective: str):
    """Run optimization and cache results."""
    _scheduler.model = None  # Reset model
    return _scheduler.optimize(objective=objective)


def create_gantt_chart(result: ScheduleResult, scheduler: GroundOpsScheduler) -> go.Figure:
    """Create an interactive Gantt chart."""
    df = result.tasks.copy()
    
    # Color mapping by mission
    mission_colors = {
        'ARTEMIS-IV': '#0B3D91',  # NASA blue
        'CLPS-2028-ALPHA': '#FC3D21',  # NASA red
        'ISS-RESUPPLY-82': '#1E88E5',  # Blue
    }
    
    fig = go.Figure()
    
    for mission_id in df['mission_id'].unique():
        mission_df = df[df['mission_id'] == mission_id]
        mission_name = scheduler.missions.get(mission_id, {}).get('name', mission_id)
        color = mission_colors.get(mission_id, '#888888')
        
        for _, row in mission_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row['duration_hours']],
                y=[row['task_name']],
                base=[row['start_hours']],
                orientation='h',
                name=mission_name,
                showlegend=False,
                marker_color=color,
                hovertemplate=(
                    f"<b>{row['task_name']}</b><br>"
                    f"Start: {row['start_time'].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"Duration: {row['duration_hours']:.1f} hours<br>"
                    f"<extra></extra>"
                )
            ))
    
    fig.update_layout(
        title="Mission Schedule Gantt Chart",
        xaxis_title="Hours from Planning Start",
        yaxis_title="Tasks",
        height=max(400, len(df) * 25),
        bargap=0.2,
        hovermode='closest'
    )
    
    return fig


def create_resource_utilization_chart(result: ScheduleResult, scheduler: GroundOpsScheduler) -> go.Figure:
    """Create resource utilization bar chart."""
    util_data = result.resource_utilization
    
    # Prepare data
    resources = []
    utilizations = []
    types = []
    
    for res_id, util in util_data.items():
        if res_id in scheduler.resources:
            res = scheduler.resources[res_id]
            resources.append(res.name)
            utilizations.append(util)
            types.append(res.type)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Resource': resources,
        'Utilization': utilizations,
        'Type': types
    }).sort_values('Utilization', ascending=True)
    
    # Color by type
    type_colors = {
        'facility': '#0B3D91',
        'crew': '#FC3D21',
        'equipment': '#1E88E5'
    }
    
    fig = go.Figure()
    
    for res_type in df['Type'].unique():
        type_df = df[df['Type'] == res_type]
        fig.add_trace(go.Bar(
            y=type_df['Resource'],
            x=type_df['Utilization'],
            orientation='h',
            name=res_type.title(),
            marker_color=type_colors.get(res_type, '#888888')
        ))
    
    fig.update_layout(
        title="Resource Utilization (%)",
        xaxis_title="Utilization (%)",
        yaxis_title="",
        height=max(300, len(df) * 30),
        barmode='group'
    )
    
    return fig


def create_timeline_chart(result: ScheduleResult, scheduler: GroundOpsScheduler) -> go.Figure:
    """Create timeline showing all missions."""
    df = result.tasks.copy()
    
    fig = make_subplots(
        rows=len(df['mission_id'].unique()),
        cols=1,
        subplot_titles=[scheduler.missions.get(m, {}).get('name', m) for m in df['mission_id'].unique()],
        vertical_spacing=0.1
    )
    
    colors = ['#0B3D91', '#FC3D21', '#1E88E5', '#4CAF50']
    
    for i, mission_id in enumerate(df['mission_id'].unique()):
        mission_df = df[df['mission_id'] == mission_id]
        
        fig.add_trace(
            go.Scatter(
                x=mission_df['start_time'],
                y=mission_df['duration_hours'],
                mode='markers+lines',
                name=mission_id,
                marker=dict(size=10, color=colors[i % len(colors)]),
                line=dict(color=colors[i % len(colors)]),
                hovertemplate='%{y:.1f} hours<extra></extra>'
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        title="Mission Timeline",
        height=200 * len(df['mission_id'].unique()),
        showlegend=False
    )
    
    return fig


def create_heatmap(result: ScheduleResult, scheduler: GroundOpsScheduler) -> go.Figure:
    """Create resource usage heatmap over time."""
    df = result.tasks.copy()
    
    # Create time buckets (days)
    df['start_day'] = df['start_hours'] // 24
    max_day = int(df['start_day'].max()) + 1
    
    # Get unique resources
    all_resources = list(scheduler.resources.keys())
    
    # Create usage matrix
    usage_matrix = np.zeros((len(all_resources), max_day + 1))
    
    for _, row in df.iterrows():
        start_day = int(row['start_day'])
        end_day = int((row['start_hours'] + row['duration_hours']) // 24)
        
        for res_id in row['resources']:
            if res_id in all_resources:
                res_idx = all_resources.index(res_id)
                for day in range(start_day, min(end_day + 1, max_day + 1)):
                    usage_matrix[res_idx, day] += row['duration_hours'] / max(1, end_day - start_day + 1)
    
    # Shorten resource names for display
    short_names = [scheduler.resources[r].name[:30] if r in scheduler.resources else r for r in all_resources]
    
    fig = go.Figure(data=go.Heatmap(
        z=usage_matrix,
        x=[f"Day {i}" for i in range(max_day + 1)],
        y=short_names,
        colorscale='Blues',
        hovertemplate='Resource: %{y}<br>Day: %{x}<br>Hours: %{z:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Resource Usage Over Time",
        xaxis_title="Day",
        yaxis_title="Resource",
        height=max(400, len(all_resources) * 25)
    )
    
    return fig


def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">🚀 NASA Ground Operations Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("**MILP-Based Scheduling Tool for Space Mission Ground Processing**")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    data_dir = st.sidebar.text_input("Data Directory", value="data")
    objective = st.sidebar.selectbox(
        "Optimization Objective",
        options=[
            "minimize_makespan",
            "minimize_cost",
            "maximize_utilization",
            "weighted"
        ],
        index=0
    )
    
    # Load data
    try:
        scheduler = load_scheduler(data_dir)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure the data directory contains missions.json, resources.json, and constraints.json")
        return
    
    # Run optimization button
    if st.sidebar.button("🔄 Run Optimization", type="primary"):
        with st.spinner("Running optimization..."):
            result = run_optimization(scheduler, objective)
            st.session_state['result'] = result
            st.session_state['scheduler'] = scheduler
    
    # Check for results
    if 'result' not in st.session_state:
        st.info("👈 Click 'Run Optimization' to start")
        
        # Show data summary
        st.subheader("📊 Data Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tasks", len(scheduler.tasks))
        col2.metric("Total Resources", len(scheduler.resources))
        col3.metric("Planning Horizon", f"{scheduler.time_horizon_hours} hours")
        
        with st.expander("View Mission Details"):
            for mission_id, mission in scheduler.missions.items():
                st.write(f"**{mission['name']}**")
                st.write(f"  - Tasks: {len(mission['tasks'])}")
                st.write(f"  - Launch Window: {mission['launch_window_start'][:10]} to {mission['launch_window_end'][:10]}")
        
        return
    
    result = st.session_state['result']
    
    # Key Metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric(
        "Makespan",
        f"{result.makespan_hours:.1f} hrs",
        f"({result.makespan_hours/24:.1f} days)"
    )
    col2.metric(
        "Total Cost",
        f"${result.total_cost:,.0f}"
    )
    col3.metric(
        "Solve Time",
        f"{result.solve_time_seconds:.2f} sec"
    )
    col4.metric(
        "Status",
        result.status.upper()
    )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Schedule", "📊 Resources", "🔬 What-If", "📋 Details"])
    
    with tab1:
        st.subheader("Gantt Chart")
        gantt_fig = create_gantt_chart(result, scheduler)
        st.plotly_chart(gantt_fig, use_container_width=True)
        
        st.subheader("Mission Timeline")
        timeline_fig = create_timeline_chart(result, scheduler)
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Resource Utilization")
            util_fig = create_resource_utilization_chart(result, scheduler)
            st.plotly_chart(util_fig, use_container_width=True)
        
        with col2:
            st.subheader("Usage Heatmap")
            heatmap_fig = create_heatmap(result, scheduler)
            st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with tab3:
        st.subheader("What-If Analysis")
        
        scenario_type = st.selectbox(
            "Select Scenario",
            options=["Custom", "Optimistic", "Pessimistic", "Resource Shortage", "Expedited"]
        )
        
        if scenario_type == "Custom":
            col1, col2 = st.columns(2)
            with col1:
                duration_mult = st.slider("Duration Multiplier", 0.5, 1.5, 1.0, 0.05)
                buffer_mult = st.slider("Safety Buffer Multiplier", 0.5, 2.0, 1.0, 0.1)
            with col2:
                launch_delay = st.number_input("Launch Window Delay (days)", 0, 30, 0)
            
            if st.button("Run Scenario"):
                params = {
                    "task_duration_multiplier": duration_mult,
                    "safety_buffer_multiplier": buffer_mult
                }
                if launch_delay > 0:
                    params["launch_window_shifts"] = {"ARTEMIS-IV": launch_delay}
                
                with st.spinner("Running scenario..."):
                    analyzer = WhatIfAnalyzer(base_data_dir=data_dir)
                    analyzer.load_base_case()
                    scenario_result = analyzer.run_scenario(params, scenario_type)
                
                st.write(scenario_result.summary())
        else:
            predefined = get_predefined_scenarios()
            scenario_key = scenario_type.lower().replace(" ", "_")
            
            if scenario_key in predefined:
                st.json(predefined[scenario_key])
                
                if st.button(f"Run {scenario_type} Scenario"):
                    with st.spinner("Running scenario..."):
                        analyzer = WhatIfAnalyzer(base_data_dir=data_dir)
                        analyzer.load_base_case()
                        scenario_result = analyzer.run_scenario(
                            predefined[scenario_key], 
                            scenario_type
                        )
                    
                    st.write(scenario_result.summary())
        
        # Sensitivity Analysis
        st.subheader("Sensitivity Analysis")
        
        sens_param = st.selectbox(
            "Parameter to Analyze",
            options=["task_duration_multiplier", "safety_buffer_multiplier"]
        )
        
        if sens_param == "task_duration_multiplier":
            values = st.slider("Range", 0.8, 1.3, (0.9, 1.2))
            sens_values = np.linspace(values[0], values[1], 5).tolist()
        else:
            values = st.slider("Range", 0.5, 2.0, (0.8, 1.5))
            sens_values = np.linspace(values[0], values[1], 5).tolist()
        
        if st.button("Run Sensitivity Analysis"):
            with st.spinner("Running sensitivity analysis..."):
                analyzer = WhatIfAnalyzer(base_data_dir=data_dir)
                analyzer.load_base_case()
                sens_result = analyzer.sensitivity_analysis(
                    parameter_name=sens_param,
                    parameter_values=sens_values
                )
            
            df = sens_result.to_dataframe()
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    go.Figure(
                        go.Scatter(x=df[sens_param], y=df['makespan_hours'], mode='lines+markers')
                    ).update_layout(title="Makespan vs Parameter", xaxis_title=sens_param, yaxis_title="Hours"),
                    use_container_width=True
                )
            with col2:
                st.plotly_chart(
                    go.Figure(
                        go.Scatter(x=df[sens_param], y=df['total_cost'], mode='lines+markers')
                    ).update_layout(title="Cost vs Parameter", xaxis_title=sens_param, yaxis_title="$"),
                    use_container_width=True
                )
    
    with tab4:
        st.subheader("Schedule Details")
        
        # Filter by mission
        mission_filter = st.multiselect(
            "Filter by Mission",
            options=result.tasks['mission_id'].unique().tolist(),
            default=result.tasks['mission_id'].unique().tolist()
        )
        
        filtered_df = result.tasks[result.tasks['mission_id'].isin(mission_filter)]
        
        # Display options
        display_cols = st.multiselect(
            "Display Columns",
            options=filtered_df.columns.tolist(),
            default=['task_name', 'start_time', 'end_time', 'duration_hours', 'hazard_level']
        )
        
        st.dataframe(
            filtered_df[display_cols].sort_values('start_time'),
            use_container_width=True
        )
        
        # Export
        if st.button("📥 Export Schedule (CSV)"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        NASA Ground Operations Optimizer | Operations Research Portfolio Project<br>
        Built with Gurobi, Streamlit, and Plotly
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
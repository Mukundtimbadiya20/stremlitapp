import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Battery Cell Monitor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .cell-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        cursor: pointer;
        transition: transform 0.3s ease;
    }

    .cell-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
    }

    .lfp-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }

    .nmc-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .status-good { color: #00ff00; font-weight: bold; }
    .status-warning { color: #ffaa00; font-weight: bold; }
    .status-critical { color: #ff0000; font-weight: bold; }

    .input-section {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def get_cell_properties(cell_type):
    """Return voltage properties for different cell chemistries"""
    properties = {
        "lfp": {
            "nominal_voltage": 3.2,
            "min_voltage": 2.8,
            "max_voltage": 3.6,
            "nominal_capacity": 100,
            "color": "#11998e"
        },
        "nmc": {
            "nominal_voltage": 3.6,
            "min_voltage": 3.2,
            "max_voltage": 4.0,
            "nominal_capacity": 80,
            "color": "#667eea"
        }
    }
    return properties.get(cell_type, properties["lfp"])


def get_cell_status(voltage, min_v, max_v, temp, current):
    """Determine cell status based on parameters"""
    if voltage < min_v or voltage > max_v or temp > 45 or abs(current) > 50:
        return "Critical", "🔴"
    elif voltage < min_v * 1.1 or voltage > max_v * 0.9 or temp > 35:
        return "Warning", "⚠️"
    else:
        return "Good", "✅"


def create_system_overview_chart(cells_data):
    """Create overview charts for all cells"""
    if not cells_data:
        return None

    df = pd.DataFrame(cells_data).T
    df['cell_name'] = df.index

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Voltage Distribution', 'Temperature vs Current',
                        'Power Output', 'Current Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Voltage distribution
    fig.add_trace(
        go.Bar(x=df['cell_name'], y=df['voltage'],
               marker_color=df['color'], name='Voltage',
               text=df['voltage'], textposition='auto'),
        row=1, col=1
    )

    # Temperature vs Current scatter
    fig.add_trace(
        go.Scatter(x=df['temp_celsius'], y=df['current'],
                   mode='markers+text',
                   marker=dict(size=15, color=df['color']),
                   text=df['cell_name'],
                   textposition='top center',
                   name='Temp vs Current'),
        row=1, col=2
    )

    # Power output
    fig.add_trace(
        go.Bar(x=df['cell_name'], y=df['power_watts'],
               marker_color=df['color'], name='Power',
               text=df['power_watts'], textposition='auto'),
        row=2, col=1
    )

    # Current distribution
    fig.add_trace(
        go.Bar(x=df['cell_name'], y=df['current'],
               marker_color=df['color'], name='Current',
               text=df['current'], textposition='auto'),
        row=2, col=2
    )

    fig.update_layout(height=600, showlegend=False,
                      title_text="Battery System Overview",
                      title_x=0.5)

    # Update axis labels
    fig.update_xaxes(title_text="Cells", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Current (A)", row=1, col=2)
    fig.update_xaxes(title_text="Cells", row=2, col=1)
    fig.update_yaxes(title_text="Power (W)", row=2, col=1)
    fig.update_xaxes(title_text="Cells", row=2, col=2)
    fig.update_yaxes(title_text="Current (A)", row=2, col=2)

    return fig


def create_detailed_cell_analysis(cell_name, cell_data):
    """Create detailed analysis charts for a specific cell"""

    # Generate realistic time series data based on current values
    hours = list(range(24))
    base_voltage = cell_data['voltage']
    base_temp = cell_data['temp_celsius']
    base_current = cell_data['current']

    # Create more realistic trends
    voltage_trend = []
    temp_trend = []
    current_trend = []

    for hour in hours:
        voltage_var = base_voltage + random.uniform(-0.05, 0.05) + (base_temp - 30) * 0.001
        voltage_trend.append(round(voltage_var, 3))
        temp_cycle = 5 * np.sin(2 * np.pi * hour / 24) + random.uniform(-1, 1)
        temp_trend.append(round(base_temp + temp_cycle, 1))
        current_var = base_current + random.uniform(-0.5, 0.5)
        current_trend.append(round(current_var, 2))

    power_trend = [round(v * c, 2) for v, c in zip(voltage_trend, current_trend)]

    # FIX: Specify subplot type for indicator
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('24h Voltage Trend', '24h Temperature Trend',
                        '24h Current Flow', '24h Power Output',
                        'Voltage vs Temperature', 'Performance Summary'),
        vertical_spacing=0.08,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"type": "indicator"}]  # <--- FIX HERE
        ]
    )

    fig.add_trace(
        go.Scatter(x=hours, y=voltage_trend, mode='lines+markers',
                   line=dict(color=cell_data['color'], width=3),
                   marker=dict(size=6),
                   name='Voltage'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=hours, y=temp_trend, mode='lines+markers',
                   line=dict(color='orange', width=3),
                   marker=dict(size=6),
                   name='Temperature'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=hours, y=current_trend, mode='lines+markers',
                   line=dict(color='red', width=3),
                   marker=dict(size=6),
                   name='Current'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=hours, y=power_trend, mode='lines+markers',
                   line=dict(color='purple', width=3),
                   marker=dict(size=6),
                   name='Power'),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=temp_trend, y=voltage_trend, mode='markers',
                   marker=dict(size=10, color=hours, colorscale='Viridis'),
                   name='V vs T'),
        row=3, col=1
    )

    efficiency = round((sum(power_trend) / len(power_trend)) / (cell_data['voltage'] * 10) * 100, 1)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=efficiency,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Efficiency %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': cell_data['color']},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}),
        row=3, col=2
    )

    fig.update_layout(height=700, showlegend=False,
                      title_text=f"🔍 Detailed Analysis: {cell_name}")

    fig.update_xaxes(title_text="Hours", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_xaxes(title_text="Hours", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_xaxes(title_text="Hours", row=2, col=1)
    fig.update_yaxes(title_text="Current (A)", row=2, col=1)
    fig.update_xaxes(title_text="Hours", row=2, col=2)
    fig.update_yaxes(title_text="Power (W)", row=2, col=2)
    fig.update_xaxes(title_text="Temperature (°C)", row=3, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=3, col=1)

    return fig


def data_input_section():
    """Create data input section"""
    st.markdown("### 📝 Battery Cell Data Input")

    if 'cells_data' not in st.session_state:
        st.session_state.cells_data = {}

    # Number of cells input
    num_cells = st.number_input("Number of cells", min_value=1, max_value=16, value=8)

    st.markdown("---")

    # Create input form
    with st.form("cell_data_form"):
        st.markdown("#### Enter data for each cell:")

        cells_data = {}

        # Create columns for input
        col1, col2 = st.columns(2)

        for i in range(num_cells):
            st.markdown(f"**Cell {i + 1}**")

            col_left, col_right = st.columns(2)

            with col_left:
                cell_type = st.selectbox(
                    f"Cell {i + 1} Type",
                    ["lfp", "nmc"],
                    key=f"type_{i}"
                )

                voltage = st.number_input(
                    f"Voltage (V)",
                    min_value=0.0, max_value=5.0, value=3.2 if cell_type == "lfp" else 3.6,
                    step=0.01, key=f"voltage_{i}"
                )

                current = st.number_input(
                    f"Current (A)",
                    min_value=-100.0, max_value=100.0, value=0.0,
                    step=0.1, key=f"current_{i}"
                )

            with col_right:
                temp = st.number_input(
                    f"Temperature (°C)",
                    min_value=-40.0, max_value=80.0, value=25.0,
                    step=0.1, key=f"temp_{i}"
                )

                soc = st.number_input(
                    f"State of Charge (%)",
                    min_value=0.0, max_value=100.0, value=50.0,
                    step=0.1, key=f"soc_{i}"
                )

            # Calculate derived values
            cell_key = f"Cell {i + 1} ({cell_type.upper()})"
            properties = get_cell_properties(cell_type)

            cells_data[cell_key] = {
                "type": cell_type,
                "voltage": voltage,
                "current": current,
                "temp_celsius": temp,
                "soc": soc,
                "nominal_capacity_ah": properties["nominal_capacity"],
                "min_voltage": properties["min_voltage"],
                "max_voltage": properties["max_voltage"],
                "color": properties["color"],
                "power_watts": round(voltage * current, 2),
                "capacity_wh": round(voltage * properties["nominal_capacity"], 2),
                "cycles": random.randint(50, 500)  # This could also be an input
            }

            st.markdown("---")

        submitted = st.form_submit_button("💾 Save Cell Data", type="primary")

        if submitted:
            st.session_state.cells_data = cells_data
            st.success(f"✅ Data saved for {num_cells} cells!")

    return st.session_state.cells_data


# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🔋 Battery Cell Monitoring Dashboard</h1>',
                unsafe_allow_html=True)

    # Initialize session state
    if 'selected_cell_for_analysis' not in st.session_state:
        st.session_state.selected_cell_for_analysis = None

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Data Input", "📱 Cell Overview", "🔍 Cell Analysis", "📊 System Analysis"])

    with tab1:
        cells_data = data_input_section()

        if cells_data:
            st.markdown("### 📋 Current Data Summary")

            # Display summary table
            summary_data = []
            for cell_name, data in cells_data.items():
                status, _ = get_cell_status(
                    data['voltage'], data['min_voltage'],
                    data['max_voltage'], data['temp_celsius'],
                    data['current']
                )

                summary_data.append({
                    'Cell': cell_name,
                    'Type': data['type'].upper(),
                    'Voltage (V)': data['voltage'],
                    'Current (A)': data['current'],
                    'Temperature (°C)': data['temp_celsius'],
                    'Power (W)': data['power_watts'],
                    'SOC (%)': data['soc'],
                    'Status': status
                })

            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)

    with tab2:
        if 'cells_data' in st.session_state and st.session_state.cells_data:
            cells_data = st.session_state.cells_data

            st.markdown("### 🔋 Cell Status Overview")
            st.markdown("*Click 'Analyze' button below any cell for detailed analysis*")

            # Display cells in a grid
            cols = st.columns(4)

            for idx, (cell_name, cell_data) in enumerate(cells_data.items()):
                with cols[idx % 4]:
                    status, status_icon = get_cell_status(
                        cell_data['voltage'], cell_data['min_voltage'],
                        cell_data['max_voltage'], cell_data['temp_celsius'],
                        cell_data['current']
                    )

                    # Cell card display
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>{status_icon} {cell_name}</h4>
                        <strong>Type:</strong> {cell_data['type'].upper()}<br>
                        <strong>Voltage:</strong> {cell_data['voltage']} V<br>
                        <strong>Current:</strong> {cell_data['current']} A<br>
                        <strong>Temperature:</strong> {cell_data['temp_celsius']} °C<br>
                        <strong>SOC:</strong> {cell_data['soc']} %<br>
                        <strong>Power:</strong> {cell_data['power_watts']} W<br>
                        <strong>Status:</strong> <span class="status-{status.lower()}">{status}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Analyze button
                    if st.button(f"🔍 Analyze", key=f"analyze_{idx}"):
                        st.session_state.selected_cell_for_analysis = cell_name
                        st.success(f"Selected {cell_name} for analysis! Go to 'Cell Analysis' tab.")
        else:
            st.warning("⚠️ Please input cell data in the 'Data Input' tab first.")

    with tab3:
        if 'cells_data' in st.session_state and st.session_state.cells_data:
            cells_data = st.session_state.cells_data

            # Cell selection for analysis
            st.markdown("### 🔍 Individual Cell Analysis")

            selected_cell = st.selectbox(
                "Select a cell for detailed analysis:",
                list(cells_data.keys()),
                index=0 if not st.session_state.selected_cell_for_analysis
                else list(cells_data.keys()).index(st.session_state.selected_cell_for_analysis)
                if st.session_state.selected_cell_for_analysis in cells_data.keys()
                else 0
            )

            if selected_cell:
                cell_info = cells_data[selected_cell]

                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Voltage", f"{cell_info['voltage']} V",
                              f"{cell_info['voltage'] - cell_info['min_voltage']:.2f} V above min")
                with col2:
                    st.metric("Current", f"{cell_info['current']} A")
                with col3:
                    st.metric("Power Output", f"{cell_info['power_watts']} W")
                with col4:
                    st.metric("SOC", f"{cell_info['soc']} %")

                # Detailed analysis charts
                detailed_fig = create_detailed_cell_analysis(selected_cell, cell_info)
                st.plotly_chart(detailed_fig, use_container_width=True)

                # Additional insights
                st.markdown("#### 📊 Cell Insights")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    **Cell Specifications:**
                    - **Chemistry:** {cell_info['type'].upper()}
                    - **Nominal Capacity:** {cell_info['nominal_capacity_ah']} Ah
                    - **Voltage Range:** {cell_info['min_voltage']} - {cell_info['max_voltage']} V
                    - **Energy Capacity:** {cell_info['capacity_wh']} Wh
                    """)

                with col2:
                    # Performance indicators
                    voltage_health = ((cell_info['voltage'] - cell_info['min_voltage']) /
                                      (cell_info['max_voltage'] - cell_info['min_voltage']) * 100)
                    temp_status = "Normal" if cell_info['temp_celsius'] < 35 else "High" if cell_info[
                                                                                                'temp_celsius'] < 45 else "Critical"

                    st.markdown(f"""
                    **Performance Status:**
                    - **Voltage Health:** {voltage_health:.1f}%
                    - **Temperature Status:** {temp_status}
                    - **Charge Level:** {cell_info['soc']:.1f}%
                    - **Power Efficiency:** {abs(cell_info['power_watts']) / cell_info['voltage']:.2f} A
                    """)
        else:
            st.warning("⚠️ Please input cell data in the 'Data Input' tab first.")

    with tab4:
        if 'cells_data' in st.session_state and st.session_state.cells_data:
            cells_data = st.session_state.cells_data

            st.markdown("### 📊 System-wide Analysis")

            # System summary
            col1, col2, col3, col4 = st.columns(4)

            total_power = sum(cell['power_watts'] for cell in cells_data.values())
            avg_temp = round(sum(cell['temp_celsius'] for cell in cells_data.values()) / len(cells_data), 1)
            avg_soc = round(sum(cell['soc'] for cell in cells_data.values()) / len(cells_data), 1)
            total_capacity = sum(cell['capacity_wh'] for cell in cells_data.values())

            with col1:
                st.metric("Total Power", f"{total_power} W")
            with col2:
                st.metric("Average Temperature", f"{avg_temp} °C")
            with col3:
                st.metric("Average SOC", f"{avg_soc} %")
            with col4:
                st.metric("Total Capacity", f"{total_capacity} Wh")

            # Overview charts
            overview_fig = create_system_overview_chart(cells_data)
            if overview_fig:
                st.plotly_chart(overview_fig, use_container_width=True)

            # Performance metrics table
            st.markdown("### 📋 Detailed Performance Table")

            df = pd.DataFrame(cells_data).T
            df_display = df[['type', 'voltage', 'current', 'temp_celsius', 'soc', 'power_watts']].copy()
            df_display.columns = ['Type', 'Voltage (V)', 'Current (A)', 'Temperature (°C)',
                                  'SOC (%)', 'Power (W)']

            st.dataframe(df_display, use_container_width=True)
        else:
            st.warning("⚠️ Please input cell data in the 'Data Input' tab first.")


if __name__ == "__main__":
    main()
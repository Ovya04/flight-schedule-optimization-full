import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os
import speech_recognition as sr
import pyttsx3
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import threading


# Import our modules
from src.data_processing.flight_data_processor import FlightDataProcessor
from src.ml_models.delay_predictor import DelayPredictor
from src.optimization.schedule_optimizer import ScheduleOptimizer
from src.optimization.cascading_impact import CascadingImpactAnalyzer
from src.utils.chatbot import FlightAnalyticsChatbot


# Page configuration
st.set_page_config(
    page_title=" Mumbai Airport Flight Optimizer",
    page_icon="flight",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Override grey text color globally to white */
    body, div, span, p, h1, h2, h3, h4, h5, h6, label {
        color: white !important;
    }
    button, input, select, textarea {
    color: initial !important;
}button:hover {
    color: black !important;
    background-color: #ffc107 !important;
}


    /* Make placeholder text white */
    ::placeholder {
        color: #ccc !important;
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)




# Enhanced Dark Theme CSS
st.markdown("""
<style>
    /* Global dark theme */
    .main, .block-container {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
   
    /* Headers in blue */
    h1, h2, h3, h4, h5, h6 {
        color: #4299e1 !important;
    }
   
    /* Buttons */
    .stButton > button {
        background-color: #1e40af !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
    }
   
    .stButton > button:hover {
        background-color: #2563eb !important;
    }
   
    /* Input fields */
    input, select, textarea {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #4299e1 !important;
    }
   
    /* Dataframes */
    .stDataFrame, .stTable {
        color: #e2e8f0 !important;
        background-color: #0f172a !important;
    }
   
    /* Sidebar */
    div[data-testid="stSidebar"] > div:first-child {
        background-color: #000000 !important;
        color: white !important;
    }
   
   
   
    .chatbot-controls {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #4299e1;
        margin-top: 10px;
    }
   
    .chat-message {
        margin: 10px 0;
        padding: 8px;
        border-radius: 8px;
        background-color: rgba(66, 153, 225, 0.1);
    }
   
    .chat-user {
        color: #80bdff;
        font-weight: bold;
    }
   
    .chat-assistant {
        color: #4299e1;
        font-weight: bold;
    }
   
    .weather-alert {
        background: linear-gradient(45deg, #ff9800, #f57c00);
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        color: white;
        font-weight: bold;
    }
   
    .runway-info {
        background: linear-gradient(45deg, #2196f3, #1976d2);
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state variables
if 'chat_messages' not in st.session_state:
    st.session_state['chat_messages'] = [
        {"role": "assistant", "content": " Welcome to Mumbai Airport Flight Operations! I'm your AI assistant ready to help you analyze flight data, delays, optimization opportunities, and much more. How can I assist you today?"}
    ]


if 'flight_data' not in st.session_state:
    st.session_state['flight_data'] = pd.DataFrame()


if 'optimization_results' not in st.session_state:
    st.session_state['optimization_results'] = pd.DataFrame()


if 'impact_analysis' not in st.session_state:
    st.session_state['impact_analysis'] = pd.DataFrame()


if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False


if 'voice_enabled' not in st.session_state:
    st.session_state['voice_enabled'] = False


if 'weather_data' not in st.session_state:
    st.session_state['weather_data'] = {}


class FlightDashboardApp:
    def __init__(self):
        # Initialize all components
        self.processor = FlightDataProcessor()
        self.predictor = DelayPredictor()
        self.optimizer = ScheduleOptimizer()
        self.impact_analyzer = CascadingImpactAnalyzer()
        self.chatbot = FlightAnalyticsChatbot()
       
        # Mumbai Airport runway specifications
        self.runway_capacity = {
            'total_runways': 2,
            'active_runways': 2,
            'max_movements_per_hour': 48,  # Combined for both runways
            'peak_capacity': 40,  # During peak hours
            'weather_reduced_capacity': 24  # During poor weather
        }


    def render_header(self):
        """Render main header"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
            <h1>Mumbai Airport Flight Schedule Optimizer</h1>
            <p><strong>AI-Powered Flight Operations Analysis & Optimization Platform</strong></p>
            <p><em>Real-time insights Weather impact Runway capacity Schedule optimization</em></p>
        </div>
        """, unsafe_allow_html=True)


    def load_and_process_data(self):
        """Load and process flight data with weather integration"""
        with st.spinner("ðŸ”„ Loading flight data and weather information..."):
            try:
                df = self.processor.fetch_and_process()
               
                if df.empty:
                    st.error("Unable to load flight data.")
                    return False
               
                # Add weather impact simulation
                df = self.simulate_weather_impact(df)
                # Add runway capacity impact
                df = self.simulate_runway_capacity_impact(df)
               
                st.session_state.flight_data = df
               
                # Train or load ML model
                if not self.predictor.load_model():
                    self.predictor.train(df)
               
                # Run optimization AFTER weather and runway analysis
                st.session_state.optimization_results = self.optimizer.optimize_schedule(df)
               
                # Analyze cascading impact with proper delay propagation
                st.session_state.impact_analysis = self.analyze_improved_cascading_impact(df)
               
                st.session_state.data_loaded = True
                st.success(f"Successfully loaded and processed {len(df)} flights!")
                return True
               
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return False


   
    def simulate_weather_impact(self, df):
        # Mock realistic weather conditions
        weather_conditions = {
            'visibility': 8.0,
            'wind_speed': 15,
            'precipitation': 'light_rain',
            'temperature': 28
        }


        st.session_state.weather_data = weather_conditions


        weather_delay = 0
        weather_reason = []


        if weather_conditions['visibility'] < 2:
            weather_delay += 30
            weather_reason.append('low_visibility')
        elif weather_conditions['visibility'] < 5:
            weather_delay += 15
            weather_reason.append('reduced_visibility')


        if weather_conditions['wind_speed'] > 30:
            weather_delay += 20
            weather_reason.append('strong_winds')
        elif weather_conditions['wind_speed'] > 20:
            weather_delay += 10
            weather_reason.append('moderate_winds')


        if weather_conditions['precipitation'] in ['heavy_rain', 'thunderstorm']:
            weather_delay += 25
            weather_reason.append('severe_weather')
        elif weather_conditions['precipitation'] in ['light_rain', 'drizzle']:
            weather_delay += 8
            weather_reason.append('light_precipitation')


        # Apply delay only to 30% random subset
        weather_affected = np.random.choice(df.index, size=int(len(df) * 0.3), replace=False)


        df['weather_delay'] = 0
        df['weather_reason'] = 'none'
        df.loc[weather_affected, 'weather_delay'] = weather_delay
        df.loc[weather_affected, 'weather_reason'] = ', '.join(weather_reason) if weather_reason else 'general_weather'


        df['departure_delay'] += df['weather_delay']


        return df


    def simulate_runway_capacity_impact(self, df):
        # Group flights by hour for capacity check
        hourly_counts = df.groupby('scheduled_hour').size()


        df['runway_delay'] = 0
        df['runway_reason'] = 'none'


        max_capacity = self.runway_capacity['max_movements_per_hour']
        weather_visibility = st.session_state.weather_data.get('visibility', 10)
        if weather_visibility < 5:
            max_capacity = self.runway_capacity['weather_reduced_capacity']


        delay_increment = 5  # reduced increment
        max_runway_delay = 60  # cap delay


        for hour, count in hourly_counts.items():
            if count > max_capacity:
                excess_flights = count - max_capacity
                flights_to_delay = df[df['scheduled_hour'] == hour].tail(excess_flights).index
                for i, idx in enumerate(flights_to_delay):
                    delay = min((i + 1) * delay_increment, max_runway_delay)
                    df.at[idx, 'runway_delay'] = delay
                    df.at[idx, 'runway_reason'] = f'runway_congestion_hour_{hour}'


        df['departure_delay'] += df['runway_delay']


        return df




    def analyze_improved_cascading_impact(self, df):
        """Improved cascading impact analysis with realistic delay propagation"""
        import networkx as nx
       
        # Create more realistic flight connections
        impact_results = []
       
        for _, flight in df.iterrows():
            base_delay = flight['departure_delay']
            weather_delay = flight.get('weather_delay', 0)
            runway_delay = flight.get('runway_delay', 0)
           
            # Calculate cascading impact based on multiple factors
            # 1. Time of day impact
            hour = flight['scheduled_hour']
            if hour in [7, 8, 9, 18, 19, 20]:  # Peak hours
                time_impact = 1.5
            else:
                time_impact = 1.0
           
            # 2. Airline network impact (flights from same airline)
            airline_flights = len(df[df['airline'] == flight.get('airline', 'Unknown')])
            airline_impact = min(airline_flights / 10, 2.0)  # Cap at 2.0
           
            # 3. Route popularity impact
            route_flights = len(df[df['to_airport'] == flight['to_airport']])
            route_impact = min(route_flights / 5, 1.8)  # Cap at 1.8
           
            # Calculate composite impact score
            impact_score = (base_delay / 60) * time_impact * airline_impact * route_impact
           
            # Simulate connections (more realistic approach)
            connections_out = max(1, int(np.random.poisson(3)))  # Poisson distribution for connections
            connections_in = max(0, int(np.random.poisson(2)))
           
            # Categorize impact
            if impact_score >= 5.0:
                impact_category = 'Very High'
            elif impact_score >= 3.0:
                impact_category = 'High'
            elif impact_score >= 1.5:
                impact_category = 'Medium'
            elif impact_score >= 0.5:
                impact_category = 'Low'
            else:
                impact_category = 'Minimal'
           
            impact_results.append({
                'flight_number': flight['flight_number'],
                'impact_score': impact_score,
                'delay_minutes': base_delay,
                'weather_delay': weather_delay,
                'runway_delay': runway_delay,
                'scheduled_hour': hour,
                'airline': flight.get('airline', 'Unknown'),
                'to_airport': flight['to_airport'],
                'connections_out': connections_out,
                'connections_in': connections_in,
                'impact_category': impact_category
            })
       
        results_df = pd.DataFrame(impact_results)
        return results_df.sort_values('impact_score', ascending=False)


    def render_performance_tab(self):
        """Tab 1: Performance Indicators"""
        df = st.session_state.flight_data
       
        st.subheader("Key Performance Indicators")
       
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
       
        total_flights = len(df)
        avg_delay = df['departure_delay'].mean()
        on_time_rate = (df['departure_delay'] <= 15).mean() * 100
        severe_delays = (df['departure_delay'] > 60).sum()
       
        with col1:
            st.metric("Total Flights", f"{total_flights:,}")
        with col2:
            st.metric("Average Delay", f"{avg_delay:.1f} min")
        with col3:
            st.metric("On-Time Performance", f"{on_time_rate:.1f}%")
        with col4:
            st.metric("Severe Delays", f"{severe_delays}")
       
        # Performance Status
        if on_time_rate >= 85:
            st.success("EXCELLENT Performance")
        elif on_time_rate >= 75:
            st.info("GOOD Performance")
        elif on_time_rate >= 65:
            st.warning("MODERATE Performance - Improvement Needed")
        else:
            st.error("POOR Performance - Immediate Action Required")
       
        # Delay Distribution Pie Chart
        st.subheader("Flight Delay Distribution")
        delay_categories = pd.cut(
            df['departure_delay'],
            bins=[-float('inf'), 0, 15, 30, 60, float('inf')],
            labels=['Early', 'On Time (0-15 min)', 'Minor Delay (16-30 min)',
                   'Major Delay (31-60 min)', 'Severe Delay (>60 min)']
        ).value_counts()
       
        fig_pie = px.pie(
            values=delay_categories.values,
            names=delay_categories.index,
            title='Flight Performance Categories',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)


    def render_timing_tab(self):
        """Tab 2: Enhanced Timing Analysis with Weather and Runway Impact"""
        df = st.session_state.flight_data
        weather_data = st.session_state.weather_data
       
        st.subheader("Weather Impact Analysis")
       
        # Weather conditions display
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Visibility", f"{weather_data.get('visibility', 10):.1f} km")
        with col2:
            st.metric("Wind Speed", f"{weather_data.get('wind_speed', 10)} km/h")
        with col3:
            st.metric("Temperature", f"{weather_data.get('temperature', 25)}Â°C")
        with col4:
            precipitation = weather_data.get('precipitation', 'clear').replace('_', ' ').title()
            st.metric("Conditions", precipitation)
       
        # Weather delay analysis
        weather_affected = df[df['weather_delay'] > 0]
        if not weather_affected.empty:
            st.markdown('<div class="weather-alert">Weather Impact Detected</div>', unsafe_allow_html=True)
           
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Flights Affected by Weather", len(weather_affected))
                st.metric("Avg Weather Delay", f"{weather_affected['weather_delay'].mean():.1f} min")
           
            with col2:
                # Weather delay distribution
                avg_weather_delay = weather_affected.groupby('scheduled_hour')['weather_delay'].mean().reset_index()


                fig_weather = px.line(
                    avg_weather_delay,
                    x='scheduled_hour',
                    y='weather_delay',
                    title='Average Weather Delay by Scheduled Hour',
                    markers=True,
                    labels={'scheduled_hour': 'Scheduled Hour', 'weather_delay': 'Avg Weather Delay (min)'}
                )


                fig_weather.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis=dict(tickmode='linear',tickfont={'color':'lightgrey'}),
    yaxis=dict(tickfont={'color':'lightgrey'}),
                )


                st.plotly_chart(fig_weather, use_container_width=True)
       
        # Runway Capacity Analysis
        st.subheader("Runway Capacity Impact")
       
        st.markdown('<div class="runway-info">Mumbai Airport Runway Specifications</div>', unsafe_allow_html=True)
       
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Runways", self.runway_capacity['active_runways'])
        with col2:
            st.metric("Max Movements/Hour", self.runway_capacity['max_movements_per_hour'])
        with col3:
            current_capacity = self.runway_capacity['weather_reduced_capacity'] if weather_data.get('visibility', 10) < 5 else self.runway_capacity['max_movements_per_hour']
            st.metric("Current Capacity", current_capacity)
        with col4:
            runway_affected = df[df['runway_delay'] > 0]
            st.metric("Runway Delays", len(runway_affected))
       
        # Hourly capacity vs demand
        hourly_flights = df.groupby('scheduled_hour').size().reset_index()
        hourly_flights.columns = ['Hour', 'Flight_Count']
       
        # Add capacity line
        fig_capacity = px.bar(
            hourly_flights,
            x='Hour',
            y='Flight_Count',
            title='Hourly Flight Demand vs Runway Capacity',
            color='Flight_Count',
            color_continuous_scale='Reds'
        )
       
        # Add capacity line
        max_cap = self.runway_capacity['max_movements_per_hour']
        fig_capacity.add_hline(
            y=max_cap,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max Capacity: {max_cap}"
        )
       
        fig_capacity.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis=dict(dtick=1),
            height=400
        )
        st.plotly_chart(fig_capacity, use_container_width=True)
       
        # Peak hours analysis (after weather and runway impact)
        st.subheader(" Optimized Peak Hours Analysis")
       
        peak_hours = hourly_flights.nlargest(3, 'Flight_Count')
        quiet_hours = hourly_flights.nsmallest(3, 'Flight_Count')
       
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Congested Hours (Post-Impact):**")
            for _, row in peak_hours.iterrows():
                exceeded = "OVER CAPACITY" if row['Flight_Count'] > max_cap else "Within Capacity"
                st.write(f" {row['Hour']:02d}:00-{row['Hour']+1:02d}:00: {row['Flight_Count']} flights {exceeded}")
       
        with col2:
            st.markdown("**Optimal Hours (Recommended):**")
            for _, row in quiet_hours.iterrows():
                available = max_cap - row['Flight_Count']
                st.write(f" {row['Hour']:02d}:00-{row['Hour']+1:02d}:00: {row['Flight_Count']} flights ({available} slots available)")


    def render_optimization_tab(self):
        """Tab 3: Enhanced Optimization & Impact Analysis"""
        opt_results = st.session_state.optimization_results
        if 'destination' in opt_results.columns:
            opt_results = opt_results.rename(columns={'destination': 'to_airport'})


        impact_results = st.session_state.impact_analysis
       
        st.subheader("New Optimized Schedule")
       
        if not opt_results.empty:
            # Enhanced optimization metrics
            col1, col2, col3, col4 = st.columns(4)
           
            total_reduction = opt_results['delay_reduction'].sum()
            avg_reduction = opt_results['delay_reduction'].mean()
            flights_improved = (opt_results['delay_reduction'] > 0).sum()
            total_time_saved = total_reduction  # in minutes
           
            with col1:
                st.metric("Total Time Saved", f"{total_time_saved:.0f} min ({total_time_saved/60:.1f} hrs)")
            with col2:
                st.metric("Average Improvement", f"{avg_reduction:.1f} min per flight")
            with col3:
                st.metric("Flights Optimized", f"{flights_improved}/{len(opt_results)}")
            with col4:
                if opt_results['original_delay'].sum() == 0:
                    efficiency_gain = 0
                else:
                    efficiency_gain = (total_reduction / opt_results['original_delay'].sum()) * 100
                    efficiency_gain = min(efficiency_gain, 73.9)  # cap below 100%


                st.metric("Efficiency Gain", f"{efficiency_gain:.1f}%")
           
            # Enhanced optimization table with actual vs scheduled times
            st.markdown("**Detailed New Schedule (Top 15 Optimizations):**")
           
            # Create enhanced display table
            display_opt = opt_results.nlargest(15, 'delay_reduction').copy()
           
            # Format times properly
            if 'original_time' in display_opt.columns and 'optimized_time' in display_opt.columns:
                display_columns = {
                    'flight_number': 'Flight Number',
                    'to_airport': 'Destination',
                    'airline': 'Airline',
                    'original_time': 'Original Schedule',
                    'optimized_time': 'New Schedule',
                    'delay_reduction': 'Time Saved (min)'
                }
               
                display_table = display_opt[list(display_columns.keys())].copy()
                display_table.columns = list(display_columns.values())
               
                # Add color coding based on time saved
                def highlight_savings(val):
                    if isinstance(val, (int, float)) and val > 30:
                        return 'background-color: #28a745; color: white'
                    elif isinstance(val, (int, float)) and val > 15:
                        return 'background-color: #ffc107; color: black'
                    return ''
               
                styled_table = display_table.style.map(highlight_savings, subset=['Time Saved (min)'])
                st.dataframe(styled_table, use_container_width=True, hide_index=True)
           
            # Download enhanced optimization results
            csv_data = opt_results.to_csv(index=False)
            st.download_button(
                label="Download Complete Optimized Schedule (CSV)",
                data=csv_data,
                file_name=f"optimized_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download the complete optimized flight schedule"
            )
       
        # Enhanced Cascading Impact Analysis
        st.subheader("Cascading Impact Analysis")
       
        if not impact_results.empty:
            # Impact summary with enhanced metrics
            high_impact = impact_results[impact_results['impact_category'].isin(['Very High', 'High'])]
           
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("High-Impact Flights", len(high_impact))
            with col2:
                avg_impact_delay = impact_results['delay_minutes'].mean()
                st.metric("Avg Delay (High-Impact)", f"{avg_impact_delay:.1f} min")
            with col3:
                total_connections = impact_results['connections_out'].sum()
                st.metric("Total Network Connections", total_connections)
            with col4:
                weather_impact_flights = len(impact_results[impact_results['weather_delay'] > 0])
                st.metric("Weather Affected", weather_impact_flights)
           
            # Enhanced high-impact flights table
            st.markdown("Critical Flights with Cascading Impact:**")
           
            display_impact = high_impact.head(15)[['flight_number', 'to_airport', 'airline',
                                                  'impact_category', 'delay_minutes', 'weather_delay',
                                                  'runway_delay', 'connections_out']].copy()
           
            display_impact.columns = ['Flight', 'Destination', 'Airline', 'Impact Level',
                                    'Total Delay (min)', 'Weather Delay (min)',
                                    'Runway Delay (min)', 'Connected Flights']
           
            st.dataframe(display_impact, use_container_width=True, hide_index=True)
           
            # Impact visualization
            fig_impact = px.scatter(
                impact_results.head(20),
                x='delay_minutes',
                y='impact_score',
                size='connections_out',
                color='impact_category',
                hover_name='flight_number',
                title='Flight Impact Analysis: Delay vs Network Impact Score',
                labels={
                    'delay_minutes': 'Total Delay (minutes)',
                    'impact_score': 'Network Impact Score',
                    'connections_out': 'Connected Flights'
                }
            )
            fig_impact.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=500
            )
            st.plotly_chart(fig_impact, use_container_width=True)
    def render_chatbot_ui(self):
   
        st.markdown("### 💬 AI Flight Assistant")


        # Display last 8 chat messages
        for msg in st.session_state.chat_messages[-8:]:
            role_class = "chat-user" if msg['role'] == 'user' else "chat-assistant"
            label = "You:" if msg['role'] == 'user' else "Assistant:"
            st.markdown(
                f'<div class="chat-message"><span class="{role_class}">{label}</span><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )


        st.markdown('<div class="chatbot-controls">', unsafe_allow_html=True)


        # Voice toggle
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("🎤", key="voice_toggle"):
                st.session_state.voice_enabled = not st.session_state.voice_enabled
        with col2:
            status = "🟢 Voice ON" if st.session_state.voice_enabled else "🔴 Voice OFF"
            st.write(status)


        # Quick suggestion buttons
        st.markdown("**💡 Quick Questions:**")
        suggestions = [
            "Weather impact summary",
            "Runway capacity status",
            "Optimization opportunities",
            "High-impact flights",
            "Performance overview"
        ]
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggest_{suggestion}", use_container_width=True):
                st.session_state['pending_chat_input'] = suggestion
                st.experimental_rerun()
                break  # handle only one per run


        # Text input
        text_input = st.text_input("💬 Ask anything about flight operations:", key="chat_input")
        if text_input and 'pending_chat_input' not in st.session_state:
            st.session_state['pending_chat_input'] = text_input
            st.experimental_rerun()


        # Voice input button
        if st.session_state.voice_enabled:
            if st.button("🎤 Speak Your Question", key="voice_input", use_container_width=True):
                voice_text = self.get_voice_input()
                if voice_text and 'pending_chat_input' not in st.session_state:
                    st.session_state['pending_chat_input'] = voice_text
                    st.experimental_rerun()


        # Export options
        st.markdown("**📥 Export Options:**")
        if len(st.session_state.chat_messages) > 1:
            chat_text = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_messages
            )
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Download TXT",
                    data=chat_text,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                pdf_buffer = self.generate_chat_pdf()
                st.download_button(
                    "📑 Download PDF",
                    data=pdf_buffer,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )


        st.markdown('</div>', unsafe_allow_html=True)


   
    def get_enhanced_chatbot_response(self, user_input,flight_data, optimization_results, impact_analysis):
        """Enhanced chatbot response with domain checking"""
        # Flight operations related keywords
        flight_keywords = [
            'flight', 'delay', 'runway', 'weather', 'airport', 'airline', 'schedule',
            'optimization', 'peak', 'hours', 'route', 'destination', 'performance',
            'impact', 'cascading', 'analysis', 'summary', 'report', 'congestion',
            'capacity', 'time', 'minutes', 'departure', 'arrival', 'operations'
        ]
       
        # Check if the question is related to flight operations
        user_lower = user_input.lower()
        is_flight_related = any(keyword in user_lower for keyword in flight_keywords)
       
        if not is_flight_related:
            return "I apologize, but that question is outside my expertise in flight operations. I specialize in analyzing flight delays, schedules, weather impacts, runway capacity, and optimization opportunities. Please ask me something related to flight operations!"
       
        # If it's flight-related, use the original chatbot
        try:
            return self.chatbot.get_response(
                user_input,flight_data,
                optimization_results,
                impact_analysis
            )
        except:
            return "I'm having trouble processing that request. Could you please rephrase your question about flight operations?"


    def process_chat_input(self, user_input):
        """Process chat input and generate response"""
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
       
        response = self.get_enhanced_chatbot_response(user_input,st.session_state.flight_data,
            st.session_state.optimization_results,
            st.session_state.impact_analysis
)
       
        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    def get_voice_input(self):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("🎤 Listening... Speak now!")
                audio = r.listen(source, timeout=5)
                text = r.recognize_google(audio)
                return text
        except:
            st.error("Could not understand audio. Please try again.")
            return None
    def generate_chat_pdf(self):
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        p.setFont("Helvetica-Bold", 16)
        p.drawString(100, height - 100, "Flight Operations Chat History")
        p.setFont("Helvetica", 10)
        p.drawString(100, height - 120, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        y_position = height - 160
        for message in st.session_state.chat_messages:
            if y_position < 100:
                p.showPage()
                y_position = height - 100

            role = message['role'].upper()
            content = message['content']

            lines = []
            words = content.split()
            current_line = f"{role}: "

            for word in words:
                if len(current_line + word) < 80:
                    current_line += word + " "
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            lines.append(current_line.strip())

            for line in lines:
                p.drawString(100, y_position, line)
                y_position -= 15

            y_position -= 10

        p.save()
        buffer.seek(0)
        return buffer


    def get_chatbot_question_bank(self):
        """Extended list of questions users can ask"""
        return {
            "Performance Analysis": [
                "What's the overall flight performance today?",
                "Show me the on-time performance statistics",
                "Which airlines have the worst delays?",
                "What percentage of flights are severely delayed?",
                "Compare airline performance rankings",
                "What's the average delay across all flights?"
            ],
            "Weather Impact": [
                "How is weather affecting flights today?",
                "Which flights are delayed due to weather?",
                "What's the visibility and wind conditions?",
                "How many flights are weather-affected?",
                "Show weather delay distribution",
                "What weather conditions cause most delays?"
            ],
            "Runway & Capacity": [
                "Are we exceeding runway capacity?",
                "Which hours have runway congestion?",
                "How many runways are active?",
                "What's the maximum hourly capacity?",
                "Show flights delayed due to runway constraints",
                "When is the best time to schedule flights?"
            ],
            "Schedule Optimization": [
                "What optimization opportunities exist?",
                "Show me the new optimized schedule",
                "How much time can we save with optimization?",
                "Which flights should be rescheduled?",
                "What's the efficiency gain from optimization?",
                "Show flights with highest optimization potential"
            ],
            "Route & Destination Analysis": [
                "Which routes have the most delays?",
                "Show me flight volume by destination",
                "What are the busiest international routes?",
                "Which destinations have best on-time performance?",
                "Analyze domestic vs international performance",
                "Show route-wise delay patterns"
            ],
            "Peak Hours & Timing": [
                "What are the peak congestion hours?",
                "When is the best time to schedule flights?",
                "Show me the quietest hours for departures",
                "Which hours should we avoid for scheduling?",
                "Analyze hourly flight distribution",
                "What's the delay pattern throughout the day?"
            ],
            "Cascading Impact": [
                "Which flights have the biggest network impact?",
                "Show me high-impact flights requiring attention",
                "How do delays cascade through the network?",
                "Which flights affect the most connections?",
                "Analyze network connectivity patterns",
                "Show flights that cause ripple effects"
            ],
            "Comprehensive Reports": [
                "Generate a comprehensive operations summary",
                "Create a detailed performance report",
                "Show me all critical issues today",
                "Provide optimization recommendations",
                "Generate executive summary",
                "Create operational insights report"
            ]
        }


    def run(self):
        """Main application runner"""
        if 'pending_chat_input' in st.session_state and st.session_state['pending_chat_input']:
            user_input = st.session_state['pending_chat_input']
            self.process_chat_input(user_input)
            st.session_state['pending_chat_input'] = None  # Clear after processing


        self.render_header()
               
        # Check if data is loaded
        if not st.session_state.data_loaded:
            st.info(" Welcome! Click the button below to load and analyze flight data with weather and runway capacity analysis.")
           
            if st.button("Load Flight Data & Start Analysis", type="primary", use_container_width=True):
                if self.load_and_process_data():
                    st.rerun()
        else:
            # Main layout: Left 75% for analysis, Right 25% for chatbot
            left_col, right_col = st.columns([3, 1])
           
            # Left column: Analysis tabs
            with left_col:
                tab1, tab2, tab3,tab4 = st.tabs([
                    "Performance Indicators",
                    " Weather & Runway Analysis",
                    "Optimization & Impact",
                    "AI Flight Assistant"
                ])
               
                with tab1:
                    self.render_performance_tab()
               
                with tab2:
                    self.render_timing_tab()
               
                with tab3:
                    self.render_optimization_tab()
                   
                with tab4:
                    self.render_chatbot_ui()
           
                                   


        # Sidebar refresh option
        with st.sidebar:
            st.header("Controls")
            if st.button("Refresh Data", help="Reload flight data with fresh weather and runway analysis"):
                st.session_state.data_loaded = False
                st.session_state.flight_data = pd.DataFrame()
                st.session_state.optimization_results = pd.DataFrame()
                st.session_state.impact_analysis = pd.DataFrame()
                st.rerun()


# Initialize and run the application
if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
   
    # Run the application
    app = FlightDashboardApp()
    app.run()
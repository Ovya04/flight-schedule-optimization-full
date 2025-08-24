import pandas as pd
import numpy as np
import pulp
import random
from datetime import datetime, timedelta

class ScheduleOptimizer:
    def __init__(self):
        self.capacity_per_slot = 3  # Max flights per 15-minute slot
        self.slot_duration = 30     # Minutes per slot
        
    def optimize_schedule(self, df):
        """Optimize flight schedule to minimize delays"""
        print("🔧 Optimizing flight schedule...")
        
        if len(df) == 0:
            print("⚠️ No flight data available for optimization")
            return pd.DataFrame()
        
        # Sample smaller dataset for optimization (performance)
        if len(df) > 30:
            df_opt = df.sample(n=30, random_state=42)
        else:
            df_opt = df.copy()
        
        # Create time slots (5 AM to 11 PM)
        slots = []
        for hour in range(5, 23):
            for minute in [0, 15, 30, 45]:
                slots.append(f"{hour:02d}:{minute:02d}")
        
        flights = df_opt.reset_index(drop=True)
        n_flights = len(flights)
        n_slots = len(slots)
        
        print(f"📊 Optimizing {n_flights} flights across {n_slots} time slots...")
        
        # Create optimization problem
        prob = pulp.LpProblem("FlightScheduleOptimization", pulp.LpMinimize)
        
        # Decision variables: x[i,j] = 1 if flight i is assigned to slot j
        x = {}
        for i in range(n_flights):
            for j in range(n_slots):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat='Binary')
        
        # Delay variables for each flight
        delay_vars = {}
        for i in range(n_flights):
            delay_vars[i] = pulp.LpVariable(f"delay_{i}", lowBound=0, cat='Continuous')
        
        # Objective: Minimize total weighted delays
        prob += pulp.lpSum([
            delay_vars[i] * (1.5 if flights.iloc[i].get('is_peak_hour', False) else 1.0)
            for i in range(n_flights)
        ])
        
        # Constraints
        
        # 1. Each flight must be assigned to exactly one slot
        for i in range(n_flights):
            prob += pulp.lpSum([x[i, j] for j in range(n_slots)]) == 1
        
        # 2. Capacity constraint: Max flights per slot
        for j in range(n_slots):
            prob += pulp.lpSum([x[i, j] for i in range(n_flights)]) <= self.capacity_per_slot
        
        # 3. Delay calculation
        for i in range(n_flights):
            original_hour = flights.iloc[i]['scheduled_hour']
            original_slot = self._find_closest_slot_index(original_hour, slots)
            
            for j in range(n_slots):
                slot_hour = int(slots[j].split(':')[0])
                time_diff = abs(slot_hour - original_hour) * 60  # Convert to minutes
                prob += delay_vars[i] >= time_diff * x[i, j]
        
        # Solve optimization
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0,timeLimit=60))
            
            if prob.status not in [pulp.LpStatusOptimal,pulp.LpStatusNotSolved]:
                print("⚠️ Optimization did not find optimal solution")
                return self._create_simple_optimization(df_opt)
                
        except Exception as e:
            print(f"❌ Optimization error: {e}")
            return self._create_simple_optimization(df_opt)
        
        # Extract results
        results = []
        total_delay_reduction = 0
        
        for i in range(n_flights):
            flight = flights.iloc[i]
            original_delay = flight.get('departure_delay', 0)
            
            # Find assigned slot
            assigned_slot = None
            optimized_delay = 0
            
            for j in range(n_slots):
                if x[i, j].varValue == 1:
                    assigned_slot = slots[j]
                    optimized_delay = delay_vars[i].varValue
                    break
            
            if assigned_slot:
                delay_reduction = max(0, original_delay - optimized_delay)
                total_delay_reduction += delay_reduction
                
                results.append({
                    'flight_number': flight['flight_number'],
                    'to_airport': flight['to_airport'],
                    'original_time': flight['STD'],
                    'optimized_time': assigned_slot,
                    'original_delay': original_delay,
                    'optimized_delay': optimized_delay,
                    'delay_reduction': delay_reduction,
                    'airline': flight.get('airline', 'Unknown')
                })
        
        results_df = pd.DataFrame(results)
        
        print(f"✅ Optimization complete!")
        print(f"   📈 Total delay reduction: {total_delay_reduction:.1f} minutes")
        print(f"   📊 Average delay reduction per flight: {total_delay_reduction/len(results):.1f} minutes")
        
        return results_df.sort_values('delay_reduction', ascending=False)
    
    def _find_closest_slot_index(self, hour, slots):
        """Find the closest slot index for a given hour"""
        target_time = f"{hour:02d}:00"
        for i, slot in enumerate(slots):
            if slot >= target_time:
                return i
        return len(slots) - 1
    
    
                # Ensure new_dt stays same day and within [5,23)
       
            
    def _create_simple_optimization(self, df):
        print("🔧 Using simple optimization approach...")

        results = []
        for _, flight in df.iterrows():
            orig_dt = flight['STD']  # pandas Timestamp

            if orig_dt.hour in [7, 8, 9, 18, 19, 20]:
                # Shift by a random number of minutes between -30 and +30
                shift_minutes = random.randint(-30, 30)
                new_dt = orig_dt + timedelta(minutes=shift_minutes)

                # Clamp new_dt hour to between 5 and 22 (5 AM to 10 PM)
                if new_dt.hour < 5:
                    new_dt = new_dt.replace(hour=5, minute=0)
                elif new_dt.hour > 22:
                    new_dt = new_dt.replace(hour=22, minute=0)

                optimized_time = new_dt.strftime("%Y-%m-%d %H:%M")

                delay_reduction = max(0, flight.get('departure_delay', 0) - abs(shift_minutes))
            else:
                optimized_time = orig_dt.strftime("%Y-%m-%d %H:%M")
                delay_reduction = 0

            results.append({
                'flight_number': flight['flight_number'],
                'to_airport': flight['to_airport'],
                'original_time': orig_dt.strftime("%Y-%m-%d %H:%M"),
                'optimized_time': optimized_time,
                'original_delay': flight.get('departure_delay', 0),
                'optimized_delay': max(0, flight.get('departure_delay', 0) - delay_reduction),
                'delay_reduction': delay_reduction,
                'airline': flight.get('airline', 'Unknown')
            })

        return pd.DataFrame(results)
        
        
    def find_best_departure_times(self, df):
        """Find the best departure times based on historical delays"""
        print("📊 Analyzing best departure times...")
        
        hourly_delays = df.groupby('scheduled_hour').agg({
            'departure_delay': ['mean', 'median', 'std', 'count']
        }).round(2)
        
        hourly_delays.columns = ['avg_delay', 'median_delay', 'delay_std', 'flight_count']
        hourly_delays = hourly_delays.reset_index()
        
        # Identify best and worst hours
        best_hours = hourly_delays.nsmallest(5, 'avg_delay')
        worst_hours = hourly_delays.nlargest(5, 'avg_delay')
        
        return {
            'hourly_analysis': hourly_delays,
            'best_hours': best_hours,
            'worst_hours': worst_hours
        }

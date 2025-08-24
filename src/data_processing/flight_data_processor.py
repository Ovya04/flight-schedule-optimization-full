import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from .api_connector import FlightAPIConnector

class FlightDataProcessor:
    def __init__(self):
        self.api = FlightAPIConnector()

    def fetch_and_process(self):
        """Generate realistic mock flight data with bounded delays and process"""
        base_date = datetime.now().date()

        airlines = {
            'AI': 'Air India',
            '6E': 'IndiGo',
            'SG': 'SpiceJet',
            'UK': 'Vistara',
            'IX': 'Air India Express',
            'QP': 'Akasa Air'
        }

        destinations = [
            'Chandigarh (IXC)', 'Hyderabad (HYD)', 'Delhi (DEL)', 'Srinagar (SXR)',
            'Bengaluru (BLR)', 'Colombo (CMB)', 'Nagpur (NAG)', 'Kolkata (CCU)',
            'Muscat (MCT)', 'Ahmedabad (AMD)', 'Dubai (DXB)', 'London (LHR)',
            'Singapore (SIN)', 'Goa (GOI)'
        ]

        aircraft_types = ['A20N', 'B38M', 'A21N', 'B738', 'A319', 'B77W']

        flights_per_day = 100
        num_days = 7
        start_hour = 6
        end_hour = 22

        flights_list = []

        for day_offset in range(num_days):
            flight_date = base_date - timedelta(days=day_offset)

            for _ in range(flights_per_day):
                airline_code = random.choice(list(airlines.keys()))
                flight_number = f"{airline_code}{random.randint(200, 9999)}"
                destination = random.choice(destinations)
                aircraft = f"{random.choice(aircraft_types)} (VT-{random.choice(['EXU', 'RTJ', 'TQB', 'RTU', 'EXK', 'EXM', 'EXL'])})"

                scheduled_hour = random.randint(start_hour, end_hour - 1)
                scheduled_minute = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
                STD = datetime.combine(flight_date, datetime.min.time()) + timedelta(hours=scheduled_hour, minutes=scheduled_minute)

                # Generate capped realistic delay: normal distribution clipped between 0 and 180 mins
                delay = np.clip(np.random.normal(15, 15), 0, 180)  # mean=15, std=15, min=0, max=180
                ATD = STD + timedelta(minutes=delay)

                departure_delay = (ATD - STD).total_seconds() / 60

                flights_list.append({
                    'flight_number': flight_number,
                    'date': flight_date.strftime('%Y-%m-%d'),
                    'from_airport': 'Mumbai (BOM)',
                    'to_airport': destination,
                    'aircraft': aircraft,
                    'STD': STD.strftime('%H:%M:%S'),
                    'ATD': ATD.strftime('%H:%M:%S'),
                    'departure_delay': departure_delay,
                    'scheduled_hour': scheduled_hour,
                    'airline_code': airline_code,
                    'airline': airlines[airline_code]
                })

        df = pd.DataFrame(flights_list)

        # Convert STD and ATD columns to datetime for accuracy
        df['STD'] = pd.to_datetime(df['date'] + ' ' + df['STD'])
        df['ATD'] = pd.to_datetime(df['date'] + ' ' + df['ATD'])

        print(f"✅ Generated mock data with {len(df)} flights across {num_days} days")

        return self.process_flight_data(df)

    def process_flight_data(self, df):
        print("⚙️ Processing flight data...")

        # Calculate accurate departure delays between datetime columns (minutes)
        df['departure_delay'] = (df['ATD'] - df['STD']).dt.total_seconds() / 60
        df['departure_delay'] = df['departure_delay'].clip(lower=0, upper=180)

        # Extract time-related features
        df['scheduled_hour'] = df['STD'].dt.hour
        df['day_of_week'] = df['STD'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])

        # Classify delay severity
        df['delay_category'] = pd.cut(
            df['departure_delay'],
            bins=[-np.inf, 0, 15, 30, 60, np.inf],
            labels=['Early', 'On Time', 'Minor Delay', 'Major Delay', 'Severe Delay']
        )

        # Classify route type
        international_codes = ['SIN', 'DXB', 'LHR', 'BKK', 'DOH', 'AUH']
        df['route_type'] = df['to_airport'].apply(
            lambda x: 'International' if any(code in str(x) for code in international_codes) else 'Domestic'
        )

        df['is_peak_hour'] = df['scheduled_hour'].isin([6, 7, 8, 9, 17, 18, 19, 20])

        print(df[['departure_delay', 'scheduled_hour']].describe())

        return df

    def _time_to_minutes(self, time_str):
        if pd.isna(time_str) or time_str == '':
            return 0
        try:
            h, m = map(int, str(time_str).split(':'))
            return h * 60 + m
        except:
            return 0

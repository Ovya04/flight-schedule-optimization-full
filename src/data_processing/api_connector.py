import requests
import pandas as pd
import random
from datetime import datetime, timedelta
from config import API_KEYS, API_ENDPOINTS

class FlightAPIConnector:
    def get_fr24_schedule(self, airport='BOM', start='', end=''):
        """Fetch flight schedule from Flightradar24 API"""
        url = API_ENDPOINTS['FR24_SCHEDULE']
        params = {
            'code': airport,
            'plugin[]': 'schedule',
            'client': 'web',
            'key': API_KEYS['FLIGHT_RADAR24'],
            'start': start,
            'end': end,
        }
        
        try:
            print(f"🔄 Fetching FR24 data for {airport} from {start} to {end}...")
            r = requests.get(url, params=params, timeout=30)
            
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}: {r.text[:200]}")
            
            if not r.text.strip():
                raise ValueError("Empty response from API")
                
            data = r.json()
            
            # Extract departures from API response
            departures = data.get('results', {}).get('schedule', {}).get('departures', [])
            
            if not departures:
                print("⚠️ No departures found in API response")
                return pd.DataFrame()
            
            flights = []
            for f in departures:
                try:
                    flights.append({
                        'flight_number': f['flight']['identification']['number']['default'],
                        'date': f['flight']['time']['scheduled']['departure'].split('T')[0],
                        'STD': f['flight']['time']['scheduled']['departure'].split('T')[1][:5],
                        'ATD': f['flight']['time']['real']['departure'].split('T')[1][:5] 
                               if f['flight']['time']['real']['departure'] else None,
                        'destination': f['flight']['airport']['destination']['code']['iata'],
                        'aircraft': f['flight']['aircraft']['model']['code'],
                        'airline': f['flight']['airline']['code']['iata'],
                    })
                except (KeyError, TypeError) as e:
                    print(f"⚠️ Skipping malformed flight record: {e}")
                    continue
            
            print(f"✅ Successfully fetched {len(flights)} flights from FR24")
            return pd.DataFrame(flights)
            
        except Exception as e:
            print(f"❌ FR24 API error: {e}")
            return pd.DataFrame()

    def get_one_week_mock_data(self):
        """Generate realistic one week of Mumbai flight data"""
        print("🎭 Generating one week of mock Mumbai flight data...")
        
        # Realistic Mumbai routes and frequencies
        routes = [
            ('Delhi (DEL)', 0.25),           # 25% of flights
            ('Bengaluru (BLR)', 0.15),       # 15% of flights
            ('Chennai (MAA)', 0.12),         # 12% of flights
            ('Kolkata (CCU)', 0.10),         # 10% of flights
            ('Hyderabad (HYD)', 0.08),       # 8% of flights
            ('Ahmedabad (AMD)', 0.06),       # 6% of flights
            ('Pune (PNQ)', 0.05),            # 5% of flights
            ('Goa (GOI)', 0.04),             # 4% of flights
            ('Cochin (COK)', 0.04),          # 4% of flights
            ('Jaipur (JAI)', 0.03),          # 3% of flights
            ('Singapore (SIN)', 0.02),       # 2% of flights
            ('Dubai (DXB)', 0.02),           # 2% of flights
            ('London (LHR)', 0.02),          # 2% of flights
            ('Bangkok (BKK)', 0.02),         # 2% of flights
        ]
        
        # Airlines operating from Mumbai
        airlines = [
            ('AI', 'Air India', 0.30),       # 30% market share
            ('6E', 'IndiGo', 0.40),          # 40% market share
            ('SG', 'SpiceJet', 0.12),        # 12% market share  
            ('UK', 'Vistara', 0.10),         # 10% market share
            ('IX', 'Air India Express', 0.05), # 5% market share
            ('QP', 'Akasa Air', 0.03),       # 3% market share
        ]
        
        # Aircraft types
        aircraft_types = [
            'A320', 'A321', 'B737', 'B738', 'A350', 'B777', 'B787', 'ATR72'
        ]
        
        flights = []
        start_date = datetime.now() - timedelta(days=7)
        
        # Generate flights for 7 days
        for day in range(7):
            current_date = start_date + timedelta(days=day)
            
            # More flights on weekdays, fewer on weekends
            if current_date.weekday() < 5:  # Monday to Friday
                daily_flights = random.randint(180, 220)  # Peak days
            else:  # Weekend
                daily_flights = random.randint(140, 180)
            
            for _ in range(daily_flights):
                # Select route based on frequency
                route_rand = random.random()
                cumulative = 0
                selected_route = routes[0][0]  # Default
                
                for route, freq in routes:
                    cumulative += freq
                    if route_rand <= cumulative:
                        selected_route = route
                        break
                
                # Select airline based on market share
                airline_rand = random.random()
                cumulative = 0
                selected_airline = airlines[0]  # Default
                
                for airline_code, airline_name, share in airlines:
                    cumulative += share
                    if airline_rand <= cumulative:
                        selected_airline = (airline_code, airline_name, share)
                        break
                
                # Generate realistic flight times
                # Peak hours: 6-9 AM, 5-8 PM
                if random.random() < 0.4:  # 40% chance peak hours
                    if random.random() < 0.6:  # Morning peak
                        hour = random.randint(6, 9)
                    else:  # Evening peak
                        hour = random.randint(17, 20)
                else:  # Off-peak hours
                    hour = random.choice([5, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23])
                
                minute = random.choice([0, 15, 30, 45])
                scheduled_time = current_date.replace(hour=hour, minute=minute, second=0)
                
                # Generate delays based on realistic patterns
                # More delays during peak hours
                if 6 <= hour <= 9 or 17 <= hour <= 20:
                    delay_minutes = max(0, random.normalvariate(25, 15))  # Higher delays in peak
                else:
                    delay_minutes = max(0, random.normalvariate(12, 10))  # Lower delays off-peak
                
                actual_time = scheduled_time + timedelta(minutes=delay_minutes)
                
                flights.append({
                    'flight_number': f"{selected_airline[0]}{random.randint(1000, 9999)}",
                    'date': current_date.strftime('%Y-%m-%d'),
                    'STD': scheduled_time.strftime('%H:%M'),
                    'ATD': actual_time.strftime('%H:%M'),
                    'destination': selected_route,
                    'aircraft': random.choice(aircraft_types),
                    'airline': selected_airline[1],
                    'airline_code': selected_airline,
                })
        
        print(f"✅ Generated {len(flights)} mock flights for 7 days")
        return pd.DataFrame(flights)

    def get_weather(self, lat, lon):
        """Get weather data for Mumbai airport"""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'appid': API_KEYS['OPENWEATHER'],
                'units': 'metric'
            }
            r = requests.get(API_ENDPOINTS['OPENWEATHER'], params=params, timeout=10)
            
            if r.status_code == 200:
                w = r.json()
                return {
                    'temp': w['main']['temp'],
                    'wind_speed': w['wind']['speed'],
                    'conditions': w['weather'][0]['main'],
                    'visibility': w.get('visibility', 10000) / 1000  # Convert to km
                }
        except Exception as e:
            print(f"⚠️ Weather API error: {e}")
        
        # Return mock weather data as fallback
        return {
            'temp': 28.5,
            'wind_speed': 12.0,
            'conditions': 'Clear',
            'visibility': 8.0
        }

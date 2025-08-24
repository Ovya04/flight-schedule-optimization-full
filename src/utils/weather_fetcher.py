# src/utils/weather_fetcher.py
from src.data_processing.api_connector import FlightAPIConnector

class WeatherFetcher:
    def __init__(self):
        self.api = FlightAPIConnector()
    def get_mumbai_weather(self):
        # Mumbai coords: 19.0896,72.8656
        return self.api.get_weather(lat=19.0896,lon=72.8656)

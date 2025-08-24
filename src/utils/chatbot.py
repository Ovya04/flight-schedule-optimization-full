# src/utils/chatbot.py
import pandas as pd
import numpy as np
from datetime import datetime

class FlightAnalyticsChatbot:
    def __init__(self):
        self.conversation_history = []
        
    def get_response(self, query, df, optimization_results=None, impact_analysis=None):
        """Generate intelligent response based on flight data analysis"""
        
        if df.empty:
            return "❌ I don't have any flight data to analyze right now. Please ensure the data is loaded correctly."
        
        query = query.lower().strip()
        self.conversation_history.append({"query": query, "timestamp": datetime.now()})
        
        try:
            # Greeting patterns
            if any(word in query for word in ['hello', 'hi', 'hey', 'greetings']):
                return "👋 Hello! I'm your Flight Operations AI Assistant. I can help you analyze flight delays, peak hours, route performance, optimization opportunities, and weather impacts. What would you like to know?"
            
            # Route analysis queries
            elif any(word in query for word in ['route', 'destination', 'where', 'city']):
                return self._analyze_routes(query, df)
            
            # Delay analysis queries
            elif any(word in query for word in ['delay', 'late', 'on time', 'punctual']):
                return self._analyze_delays(query, df)
            
            # Time and schedule queries
            elif any(word in query for word in ['time', 'hour', 'schedule', 'when', 'peak', 'busy', 'free', 'slot']):
                return self._analyze_timing(query, df)
            
            # Airline performance queries
            elif any(word in query for word in ['airline', 'carrier', 'best airline', 'worst airline']):
                return self._analyze_airlines(query, df)
            
            # Optimization queries
            elif any(word in query for word in ['optim', 'improve', 'better', 'recommend', 'suggest']):
                return self._provide_optimization_insights(query, df, optimization_results)
            
            # Impact analysis queries
            elif any(word in query for word in ['impact', 'cascade', 'affect', 'influence']):
                return self._analyze_impact(query, df, impact_analysis)
            
            # Summary queries
            elif any(word in query for word in ['summary', 'overview', 'report', 'insights']):
                return self._generate_comprehensive_summary(df, optimization_results, impact_analysis)
            
            # Weather impact queries
            elif any(word in query for word in ['weather', 'rain', 'wind', 'visibility']):
                return self._analyze_weather_impact(query, df)
            
            # General statistics
            else:
                return self._provide_general_statistics(df)
                
        except Exception as e:
            return f"❌ I encountered an error analyzing the data: {str(e)}. Please try rephrasing your question."
    
    def _analyze_routes(self, query, df):
        if 'destination' in df.columns:
            route_analysis = df.groupby('destination').agg({
                'departure_delay': ['mean', 'count'],
                'flight_number': 'count'
            }).round(2)
            
            route_analysis.columns = ['avg_delay', 'delay_count', 'total_flights']
            route_analysis = route_analysis.sort_values('total_flights', ascending=False)
            
            response = "🛫 **Top 5 Routes from Mumbai:**\n\n"
            for dest, stats in route_analysis.head(5).iterrows():
                response += f"✈️ **{dest}**: {int(stats['total_flights'])} flights, {stats['avg_delay']:.1f} min avg delay\n"
            
            return response
        return "❌ Route information is not available in the current dataset."
    
    def _analyze_delays(self, query, df):
        if 'departure_delay' not in df.columns:
            return "❌ Delay information is not available in the current dataset."
        
        avg_delay = df['departure_delay'].mean()
        on_time_rate = (df['departure_delay'] <= 15).mean() * 100
        
        response = f"⏰ **Flight Delay Analysis:**\n\n"
        response += f"📊 Average delay: {avg_delay:.1f} minutes\n"
        response += f"🎯 On-time rate: {on_time_rate:.1f}%\n\n"
        
        if on_time_rate >= 80:
            response += "✅ **Assessment:** Excellent performance!"
        elif on_time_rate >= 70:
            response += "🟡 **Assessment:** Good performance with room for improvement."
        else:
            response += "🔴 **Assessment:** Poor performance. Optimization needed!"
        
        return response
    
    def _analyze_timing(self, query, df):
        if 'scheduled_hour' not in df.columns:
            return "❌ Timing information is not available."
        
        hourly_stats = df.groupby('scheduled_hour').agg({
            'departure_delay': 'mean',
            'flight_number': 'count'
        }).round(2)
        
        peak_hours = hourly_stats.nlargest(3, 'flight_number')
        quiet_hours = hourly_stats.nsmallest(3, 'flight_number')
        
        if 'busy' in query or 'peak' in query:
            response = "🕐 **Peak Hours (Busiest Times):**\n\n"
            for hour, stats in peak_hours.iterrows():
                response += f"⏰ **{hour:02d}:00-{hour+1:02d}:00**: {int(stats['flight_number'])} flights\n"
        elif 'free' in query or 'quiet' in query:
            response = "✅ **Quietest Hours (Best for Scheduling):**\n\n"
            for hour, stats in quiet_hours.iterrows():
                response += f"🟢 **{hour:02d}:00-{hour+1:02d}:00**: {int(stats['flight_number'])} flights\n"
        else:
            response = f"🕐 **Peak Hours:** {', '.join([f'{h:02d}:00' for h in peak_hours.index])}\n"
            response += f"🟢 **Quiet Hours:** {', '.join([f'{h:02d}:00' for h in quiet_hours.index])}"
        
        return response
    
    def _analyze_airlines(self, query, df):
        if 'airline' not in df.columns:
            return "❌ Airline information not available."
        
        airline_stats = df.groupby('airline').agg({
            'departure_delay': 'mean',
            'flight_number': 'count'
        }).round(1)
        
        best_airline = airline_stats.loc[airline_stats['departure_delay'].idxmin()]
        worst_airline = airline_stats.loc[airline_stats['departure_delay'].idxmax()]
        
        response = f"✈️ **Airline Performance:**\n\n"
        response += f"🏆 **Best:** {best_airline.name} ({best_airline['departure_delay']:.1f} min avg delay)\n"
        response += f"📉 **Worst:** {worst_airline.name} ({worst_airline['departure_delay']:.1f} min avg delay)"
        
        return response
    
    def _provide_optimization_insights(self, query, df, optimization_results):
        response = "🚀 **Schedule Optimization Insights:**\n\n"
        
        if optimization_results is not None and not optimization_results.empty:
            total_reduction = optimization_results['delay_reduction'].sum()
            response += f"📈 **Total delay reduction possible:** {total_reduction:.1f} minutes\n"
            response += f"🎯 **Flights that can be optimized:** {len(optimization_results)}\n\n"
            response += "💡 **Key recommendations:**\n"
            response += "• Move peak-hour flights to off-peak slots\n"
            response += "• Implement slot-based scheduling\n"
            response += "• Focus on high-impact flights first"
        else:
            response += "📊 Run optimization analysis to see improvement opportunities."
        
        return response
    
    def _analyze_impact(self, query, df, impact_analysis):
        response = "🔗 **Cascading Impact Analysis:**\n\n"
        
        if impact_analysis is not None and not impact_analysis.empty:
            high_impact = impact_analysis[impact_analysis['impact_category'].isin(['Very High', 'High'])]
            response += f"⚠️ **High-impact flights:** {len(high_impact)}\n"
            response += f"📊 **Total flights analyzed:** {len(impact_analysis)}\n\n"
            response += "💡 **Priority actions:**\n"
            response += "• Monitor high-impact flights closely\n"
            response += "• Implement contingency plans\n"
            response += "• Prioritize ground operations for these flights"
        else:
            response += "📊 Run impact analysis to see flight connections."
        
        return response
    
    def _analyze_weather_impact(self, query, df):
        response = "🌤️ **Weather Impact on Flight Operations:**\n\n"
        response += "☁️ **Common weather factors:**\n"
        response += "• **Visibility:** Low visibility causes 15-30 min delays\n"
        response += "• **Wind:** Strong crosswinds add 10-20 min delays\n"
        response += "• **Rain:** Heavy precipitation causes 20-45 min delays\n\n"
        response += "💡 **Mitigation strategies:**\n"
        response += "• Monitor forecasts 6-12 hours ahead\n"
        response += "• Use ground delay programs during severe weather\n"
        response += "• Maintain alternate airport options"
        
        return response
    
    def _generate_comprehensive_summary(self, df, optimization_results, impact_analysis):
        total_flights = len(df)
        avg_delay = df['departure_delay'].mean() if 'departure_delay' in df.columns else 0
        on_time_rate = (df['departure_delay'] <= 15).mean() * 100 if 'departure_delay' in df.columns else 0
        
        response = "📋 **Comprehensive Flight Operations Summary**\n"
        response += "=" * 50 + "\n\n"
        response += f"📊 **Overall Performance:**\n"
        response += f"• Total flights: {total_flights:,}\n"
        response += f"• Average delay: {avg_delay:.1f} minutes\n"
        response += f"• On-time performance: {on_time_rate:.1f}%\n\n"
        
        if on_time_rate >= 80:
            response += "✅ **Status:** EXCELLENT Performance\n"
        elif on_time_rate >= 70:
            response += "🟡 **Status:** GOOD Performance\n"
        else:
            response += "🔴 **Status:** POOR Performance - Action Required\n"
        
        return response
    
    def _provide_general_statistics(self, df):
        response = f"📊 **Flight Operations Statistics:**\n\n"
        response += f"✈️ **Dataset Overview:**\n"
        response += f"• Total flights: {len(df):,}\n"
        
        if 'departure_delay' in df.columns:
            avg_delay = df['departure_delay'].mean()
            response += f"• Average delay: {avg_delay:.1f} minutes\n"
        
        response += f"\n💬 **Try asking me about:**\n"
        response += f"• 'Show me peak hours'\n"
        response += f"• 'Which airlines perform best?'\n"
        response += f"• 'How can I reduce delays?'\n"
        response += f"• 'Generate a summary report'"
        
        return response

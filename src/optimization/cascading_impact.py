import pandas as pd
import networkx as nx
import numpy as np
import streamlit as st

class CascadingImpactAnalyzer:
    def analyze_cascading_impact(self, df):
        """Analyze which flights have the biggest cascading impact on delays"""
        print("🔗 Analyzing cascading impact of flight delays...")
        
        if len(df) < 5:
            print("⚠️ Insufficient data for cascading analysis")
            return pd.DataFrame()
        
        # Create flight network graph
        G = nx.DiGraph()
        
        # Add nodes (flights) with their properties
        for _, flight in df.iterrows():
            G.add_node(
                flight['flight_number'],
                delay=flight.get('departure_delay', 0),
                hour=flight.get('scheduled_hour', 12),
                airline=flight.get('airline', 'Unknown'),
                destination=flight.get('to_airport', 'Unknown')
            )
        
        # Add edges based on potential connections
        flights_list = df.to_dict('records')
        
        for i, flight1 in enumerate(flights_list):
            for j, flight2 in enumerate(flights_list):
                if i != j:
                    # Connect flights that could impact each other
                    # Same airline (crew/aircraft rotation)
                    if flight1.get('airline') == flight2.get('airline'):
                        time_diff = abs(flight1.get('scheduled_hour', 0) - flight2.get('scheduled_hour', 0))
                        if 1 <= time_diff <= 4:  # 1-4 hours apart
                            weight = flight1.get('departure_delay', 0) / (time_diff + 1)
                            G.add_edge(flight1['flight_number'], flight2['flight_number'], weight=weight)
                    
                    # Same time slot (runway congestion)
                    if abs(flight1.get('scheduled_hour', 0) - flight2.get('scheduled_hour', 0)) <= 1:
                        if flight1.get('departure_delay', 0) > 15:  # Only if significantly delayed
                            G.add_edge(flight1['flight_number'], flight2['flight_number'], 
                                     weight=flight1.get('departure_delay', 0) * 0.5)
        
        if len(G.edges()) == 0:
            print("⚠️ No flight connections found for cascading analysis")
            return self._create_simple_impact_analysis(df)
        
        # Calculate centrality measures
        try:
            pagerank = nx.pagerank(G, weight='weight')
            betweenness = nx.betweenness_centrality(G, weight='weight')
            in_degree = dict(G.in_degree(weight='weight'))
            out_degree = dict(G.out_degree(weight='weight'))
        except:
            print("⚠️ Error calculating network metrics, using simple analysis")
            return self._create_simple_impact_analysis(df)
        
        # Create impact analysis results
        impact_results = []
        
        for flight_num in G.nodes():
            flight_data = G.nodes[flight_num]
            
            # Calculate composite impact score
            impact_score = (
                pagerank.get(flight_num, 0) * 0.4 +
                betweenness.get(flight_num, 0) * 0.3 +
                (out_degree.get(flight_num, 0) / 100) * 0.3
            )
            st.write(flight_data.columns)
            impact_results.append({
                'flight_number': flight_num,
                'impact_score': impact_score,
                'delay_minutes': flight_data['delay'],
                'scheduled_hour': flight_data['hour'],
                'airline': flight_data['airline'],
                
                'destination': flight_data['to_airport'],
                'connections_out': G.out_degree(flight_num),
                'connections_in': G.in_degree(flight_num),
                'impact_category': self._categorize_impact(impact_score)
            })
        
        results_df = pd.DataFrame(impact_results)
        results_df = results_df.sort_values('impact_score', ascending=False)
        
        print(f"✅ Cascading impact analysis complete for {len(results_df)} flights")
        
        return results_df.head(20)  # Return top 20 high-impact flights
    
    def _categorize_impact(self, score):
        """Categorize impact score into levels"""
        if score >= 0.1:
            return 'Very High'
        elif score >= 0.05:
            return 'High'
        elif score >= 0.02:
            return 'Medium'
        elif score >= 0.01:
            return 'Low'
        else:
            return 'Minimal'
    
    def _create_simple_impact_analysis(self, df):
        """Simple impact analysis when network analysis fails"""
        print("🔧 Using simple impact analysis...")
        
        # Simple heuristic: flights with high delays during peak hours have high impact
        impact_results = []
        
        for _, flight in df.iterrows():
            delay = flight.get('departure_delay', 0)
            hour = flight.get('scheduled_hour', 12)
            
            # Base impact score on delay and timing
            if hour in [7, 8, 9, 18, 19, 20]:  # Peak hours
                impact_multiplier = 1.5
            else:
                impact_multiplier = 1.0
            
            impact_score = (delay / 60) * impact_multiplier  # Normalize by hour
            
            impact_results.append({
                'flight_number': flight['flight_number'],
                'impact_score': impact_score,
                'delay_minutes': delay,
                'scheduled_hour': hour,
                'airline': flight.get('airline', 'Unknown'),
                'destination': flight.get('to_airport', 'Unknown'),
                'connections_out': 0,
                'connections_in': 0,
                'impact_category': self._categorize_impact(impact_score)
            })
        
        results_df = pd.DataFrame(impact_results)
        return results_df.sort_values('impact_score', ascending=False).head(20)

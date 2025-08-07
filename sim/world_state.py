import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import networkx as nx
from .entities import Town, Caravan, Item

class WorldState:
    """Manages the global state of the economic simulation"""
    
    def __init__(self):
        self.towns: Dict[str, Town] = {}
        self.caravans: Dict[str, Caravan] = {}
        self.items: Dict[str, Item] = {}
        self.trade_network = nx.Graph()
        self.current_time = datetime.now()
        self.market_events: List[Dict] = []
        
        # Economic indicators
        self.global_gdp = 0.0
        self.global_trade_volume = 0.0
        self.price_indices: Dict[str, List[float]] = {}
        
        # Load initial data
        self._load_towns()
        self._load_items()
        self._build_trade_network()
        
    def _load_towns(self):
        """Load towns from CSV data"""
        try:
            df = pd.read_csv('data/towns.csv')
            for _, row in df.iterrows():
                town = Town(
                    country=row['country'],
                    region=row['region'],
                    population=row['population'],
                    gdp_per_capita=row['gdp_per_capita'],
                    tech_level=row['tech_level'],
                    stability=row['stability'],
                    latitude=row['latitude'],
                    longitude=row['longitude']
                )
                self.towns[row['country']] = town
        except FileNotFoundError:
            print("Warning: towns.csv not found, using default towns")
            self._create_default_towns()
    
    def _create_default_towns(self):
        """Create default towns if CSV is not available"""
        default_towns = [
            ('USA', 'North America', 331002651, 69287, 9, 8, 39.8283, -98.5795),
            ('China', 'Asia', 1439323776, 12556, 8, 7, 35.8617, 104.1954),
            ('Germany', 'Europe', 83783942, 51216, 9, 9, 51.1657, 10.4515),
            ('Japan', 'Asia', 125836021, 40113, 9, 9, 36.2048, 138.2529),
            ('India', 'Asia', 1380004385, 2100, 6, 6, 20.5937, 78.9629)
        ]
        
        for country, region, pop, gdp, tech, stab, lat, lon in default_towns:
            town = Town(country, region, pop, gdp, tech, stab, lat, lon)
            self.towns[country] = town
    
    def _load_items(self):
        """Load items from CSV data"""
        try:
            df = pd.read_csv('data/items.csv')
            for _, row in df.iterrows():
                item = Item(
                    item_type=row['item_type'],
                    base_price=row['base_price'],
                    volatility=row['volatility'],
                    weight=row['weight'],
                    perishable=row['perishable'],
                    tech_requirement=row['tech_requirement'],
                    demand_elasticity=row['demand_elasticity']
                )
                self.items[row['item_type']] = item
        except FileNotFoundError:
            print("Warning: items.csv not found, using default items")
            self._create_default_items()
    
    def _create_default_items(self):
        """Create default items if CSV is not available"""
        default_items = [
            ('Food', 100, 0.3, 1.0, True, 1, 0.8),
            ('Raw Resource', 50, 0.4, 2.0, False, 1, 0.6),
            ('Tech', 500, 0.6, 0.5, False, 7, 1.2),
            ('Armaments', 300, 0.7, 1.5, False, 6, 0.9),
            ('Refined Resources', 200, 0.5, 1.0, False, 4, 0.7),
            ('Energies', 150, 0.8, 0.8, False, 3, 1.1),
            ('Delicacies', 800, 0.9, 0.3, True, 2, 1.5),
            ('Medicine', 400, 0.4, 0.2, True, 5, 0.3)
        ]
        
        for item_type, base_price, volatility, weight, perishable, tech_req, demand_el in default_items:
            item = Item(item_type, base_price, volatility, weight, perishable, tech_req, demand_el)
            self.items[item_type] = item
    
    def _build_trade_network(self):
        """Build the trade network graph"""
        # Add nodes (towns)
        for town in self.towns.values():
            self.trade_network.add_node(town.country, town=town)
        
        # Add edges (trade routes) based on distance and economic factors
        for origin in self.towns.values():
            for destination in self.towns.values():
                if origin != destination:
                    distance = origin.get_distance_to(destination)
                    
                    # Only connect towns within reasonable distance or with strong economic ties
                    if distance < 8000 or self._should_connect_economically(origin, destination):
                        weight = self._calculate_route_weight(origin, destination, distance)
                        self.trade_network.add_edge(origin.country, destination.country, 
                                                   weight=weight, distance=distance)
    
    def _should_connect_economically(self, origin: Town, destination: Town) -> bool:
        """Determine if two towns should be connected based on economic factors"""
        # Connect major economies
        major_economies = ['USA', 'China', 'Germany', 'Japan', 'UK', 'France']
        if origin.country in major_economies and destination.country in major_economies:
            return True
        
        # Connect regional neighbors
        if origin.region == destination.region:
            return True
        
        # Connect high-tech countries
        if origin.tech_level >= 8 and destination.tech_level >= 8:
            return True
        
        return False
    
    def _calculate_route_weight(self, origin: Town, destination: Town, distance: float) -> float:
        """Calculate the weight/strength of a trade route"""
        # Base weight based on distance (inverse relationship)
        base_weight = 1.0 / (1.0 + distance / 1000.0)
        
        # Economic factor
        economic_factor = (origin.gdp_per_capita + destination.gdp_per_capita) / 100000.0
        
        # Stability factor
        stability_factor = (origin.stability + destination.stability) / 20.0
        
        # Tech compatibility
        tech_factor = 1.0 - abs(origin.tech_level - destination.tech_level) / 10.0
        
        return base_weight * economic_factor * stability_factor * tech_factor
    
    def get_available_routes(self, origin_country: str) -> List[Tuple[str, float, float]]:
        """Get available trade routes from a country"""
        if origin_country not in self.towns:
            return []
        
        routes = []
        origin = self.towns[origin_country]
        
        for destination_country in self.towns.keys():
            if destination_country != origin_country:
                if self.trade_network.has_edge(origin_country, destination_country):
                    edge_data = self.trade_network[origin_country][destination_country]
                    distance = edge_data['distance']
                    weight = edge_data['weight']
                    routes.append((destination_country, distance, weight))
        
        return sorted(routes, key=lambda x: x[2], reverse=True)  # Sort by weight
    
    def create_caravan(self, origin_country: str, destination_country: str) -> Optional[Caravan]:
        """Create a new caravan between two countries"""
        if origin_country not in self.towns or destination_country not in self.towns:
            return None
        
        origin = self.towns[origin_country]
        destination = self.towns[destination_country]
        
        caravan_id = f"caravan_{len(self.caravans)}_{origin_country}_{destination_country}"
        caravan = Caravan(caravan_id, origin, destination)
        
        self.caravans[caravan_id] = caravan
        return caravan
    
    def update_world_state(self, days_elapsed: int = 1):
        """Update the world state for the given number of days"""
        self.current_time += timedelta(days=days_elapsed)
        
        # Update town prices
        for town in self.towns.values():
            town.update_prices(self.market_events)
        
        # Update caravan progress
        for caravan in self.caravans.values():
            if caravan.is_active:
                caravan.update_progress(days_elapsed)
        
        # Update global indicators
        self._update_global_indicators()
        
        # Clear old market events
        self._cleanup_market_events()
    
    def _update_global_indicators(self):
        """Update global economic indicators"""
        # Calculate global GDP
        self.global_gdp = sum(town.population * town.gdp_per_capita for town in self.towns.values())
        
        # Calculate global trade volume (simplified)
        self.global_trade_volume = sum(caravan.cargo_value for caravan in self.caravans.values())
        
        # Update price indices
        for item_type in self.items.keys():
            if item_type not in self.price_indices:
                self.price_indices[item_type] = []
            
            # Calculate average price across all towns
            prices = [town.prices.get(item_type, 0) for town in self.towns.values()]
            avg_price = sum(prices) / len(prices) if prices else 0
            self.price_indices[item_type].append(avg_price)
    
    def _cleanup_market_events(self):
        """Remove old market events"""
        current_time = self.current_time
        self.market_events = [
            event for event in self.market_events 
            if 'expiry_time' not in event or event['expiry_time'] > current_time
        ]
    
    def add_market_event(self, event_type: str, item_type: str, country: str, 
                        price_multiplier: float, duration_days: int = 30):
        """Add a market event that affects prices"""
        event = {
            'type': event_type,
            'item_type': item_type,
            'country': country,
            'price_multiplier': price_multiplier,
            'start_time': self.current_time,
            'expiry_time': self.current_time + timedelta(days=duration_days)
        }
        self.market_events.append(event)
    
    def get_town_info(self, country: str) -> Optional[Dict]:
        """Get comprehensive information about a town"""
        if country not in self.towns:
            return None
        
        town = self.towns[country]
        return {
            'country': town.country,
            'region': town.region,
            'population': town.population,
            'gdp_per_capita': town.gdp_per_capita,
            'tech_level': town.tech_level,
            'stability': town.stability,
            'inventory': town.inventory.copy(),
            'prices': town.prices.copy(),
            'production_capabilities': town.production_capabilities.copy()
        }
    
    def get_global_statistics(self) -> Dict:
        """Get global economic statistics"""
        return {
            'total_towns': len(self.towns),
            'active_caravans': len([c for c in self.caravans.values() if c.is_active]),
            'global_gdp': self.global_gdp,
            'global_trade_volume': self.global_trade_volume,
            'current_time': self.current_time,
            'active_market_events': len(self.market_events)
        }
    
    def save_trade_log(self, trade_data: Dict):
        """Save trade data to CSV"""
        try:
            df = pd.read_csv('data/trade_logs.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['timestamp', 'origin', 'destination', 'item_type', 
                                     'quantity', 'price_per_unit', 'total_value', 
                                     'caravan_id', 'risk_level', 'travel_time'])
        
        # Add new trade data
        new_row = pd.DataFrame([trade_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save to CSV
        df.to_csv('data/trade_logs.csv', index=False) 
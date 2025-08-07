import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

@dataclass
class Item:
    """Represents a tradeable item in the economy"""
    item_type: str
    base_price: float
    volatility: float
    weight: float
    perishable: bool
    tech_requirement: int
    demand_elasticity: float
    
    def __post_init__(self):
        self.current_price = self.base_price
        self.price_history = [self.base_price]
        self.supply = 1000  # Base supply
        self.demand = 1000  # Base demand

class Town:
    """Represents a country/town in the economic simulation"""
    
    def __init__(self, country: str, region: str, population: int, gdp_per_capita: float,
                 tech_level: int, stability: int, latitude: float, longitude: float):
        self.country = country
        self.region = region
        self.population = population
        self.gdp_per_capita = gdp_per_capita
        self.tech_level = tech_level
        self.stability = stability
        self.latitude = latitude
        self.longitude = longitude
        
        # Economic state
        self.inventory: Dict[str, int] = {}
        self.prices: Dict[str, float] = {}
        self.demand: Dict[str, float] = {}
        self.supply: Dict[str, float] = {}
        
        # Reputation system
        self.reputation: Dict[str, float] = {}  # Reputation with other countries (-1.0 to 1.0)
        self.allies: List[str] = []
        self.enemies: List[str] = []
        
        # Production focus system
        self.production_focus: Dict[str, float] = {}  # Focus multipliers for different item types
        self.specialization_bonus: float = 1.0  # Bonus for specialized production
        
        # Production capabilities (based on country characteristics)
        self.production_capabilities = self._calculate_production_capabilities()
        
        # Initialize reputation and production focus
        self._initialize_reputation()
        self._initialize_production_focus()
        
        # Initialize inventory and prices
        self._initialize_economy()
    
    def _initialize_reputation(self):
        """Initialize reputation with other countries based on historical relations"""
        # Default reputation is neutral (0.0)
        default_reputation = 0.0
        
        # Historical alliances and conflicts
        alliances = {
            'USA': ['UK', 'Canada', 'Germany', 'France', 'Japan', 'South Korea', 'Australia'],
            'UK': ['USA', 'Canada', 'Australia', 'Germany', 'France'],
            'Germany': ['USA', 'UK', 'France', 'Italy', 'Spain', 'Netherlands', 'Switzerland'],
            'France': ['USA', 'UK', 'Germany', 'Italy', 'Spain'],
            'China': ['Russia', 'North Korea'],
            'Russia': ['China', 'Belarus', 'Kazakhstan'],
            'Japan': ['USA', 'South Korea', 'Australia'],
            'South Korea': ['USA', 'Japan', 'Australia'],
            'Canada': ['USA', 'UK', 'Australia'],
            'Australia': ['USA', 'UK', 'Canada', 'Japan', 'South Korea'],
            'Italy': ['Germany', 'France', 'Spain'],
            'Spain': ['Germany', 'France', 'Italy'],
            'Netherlands': ['Germany', 'Belgium', 'Luxembourg'],
            'Switzerland': ['Germany', 'France', 'Italy', 'Austria'],
            'Saudi Arabia': ['USA', 'UAE', 'Kuwait'],
            'Turkey': ['USA', 'Germany', 'France'],
            'Brazil': ['Argentina', 'Chile', 'Uruguay'],
            'India': ['USA', 'UK', 'France'],
            'Mexico': ['USA', 'Canada'],
            'Indonesia': ['Malaysia', 'Singapore', 'Thailand']
        }
        
        enemies = {
            'USA': ['Russia', 'China', 'Iran', 'North Korea'],
            'UK': ['Russia', 'Iran'],
            'Germany': ['Russia'],
            'France': ['Russia'],
            'China': ['USA', 'Japan', 'India'],
            'Russia': ['USA', 'UK', 'Germany', 'France', 'Ukraine'],
            'Japan': ['China', 'North Korea'],
            'South Korea': ['North Korea'],
            'India': ['China', 'Pakistan'],
            'Turkey': ['Syria', 'Greece'],
            'Saudi Arabia': ['Iran', 'Qatar'],
            'Brazil': ['Venezuela'],
            'Mexico': ['Venezuela'],
            'Indonesia': ['Malaysia']
        }
        
        # Initialize reputation for all countries
        all_countries = [
            'USA', 'China', 'Germany', 'Japan', 'India', 'Brazil', 'Russia', 'France', 
            'UK', 'Canada', 'Australia', 'South Korea', 'Italy', 'Spain', 'Mexico', 
            'Indonesia', 'Netherlands', 'Saudi Arabia', 'Turkey', 'Switzerland'
        ]
        
        for country in all_countries:
            if country == self.country:
                self.reputation[country] = 1.0  # Self-reputation is always maximum
            else:
                # Check if ally
                if country in alliances.get(self.country, []):
                    self.reputation[country] = 0.7 + np.random.uniform(0.0, 0.2)
                    self.allies.append(country)
                # Check if enemy
                elif country in enemies.get(self.country, []):
                    self.reputation[country] = -0.5 + np.random.uniform(-0.3, 0.1)
                    self.enemies.append(country)
                # Neutral
                else:
                    self.reputation[country] = default_reputation + np.random.uniform(-0.1, 0.1)
    
    def _initialize_production_focus(self):
        """Initialize production focus based on country characteristics"""
        # Base focus is 1.0 for all items
        item_types = ['Food', 'Raw Resource', 'Tech', 'Armaments', 'Refined Resources', 
                     'Energies', 'Delicacies', 'Medicine']
        
        for item_type in item_types:
            self.production_focus[item_type] = 1.0
        
        # Specialize based on country characteristics
        if self.country == 'USA':
            # Tech and Armaments focus
            self.production_focus['Tech'] = 2.0
            self.production_focus['Armaments'] = 2.0
            self.production_focus['Medicine'] = 1.5
        elif self.country == 'China':
            # Manufacturing and Tech focus
            self.production_focus['Tech'] = 1.8
            self.production_focus['Refined Resources'] = 1.8
            self.production_focus['Raw Resource'] = 1.5
        elif self.country == 'Germany':
            # Tech and Refined Resources focus
            self.production_focus['Tech'] = 1.8
            self.production_focus['Refined Resources'] = 1.8
            self.production_focus['Armaments'] = 1.5
        elif self.country == 'Japan':
            # Tech and Delicacies focus
            self.production_focus['Tech'] = 2.0
            self.production_focus['Delicacies'] = 1.8
            self.production_focus['Medicine'] = 1.5
        elif self.country == 'Russia':
            # Raw Resources and Energies focus
            self.production_focus['Raw Resource'] = 2.0
            self.production_focus['Energies'] = 2.0
            self.production_focus['Armaments'] = 1.5
        elif self.country == 'Saudi Arabia':
            # Energies focus
            self.production_focus['Energies'] = 2.5
            self.production_focus['Raw Resource'] = 1.5
        elif self.country == 'Brazil':
            # Food and Raw Resources focus
            self.production_focus['Food'] = 1.8
            self.production_focus['Raw Resource'] = 1.8
            self.production_focus['Delicacies'] = 1.5
        elif self.country == 'India':
            # Food and Tech focus
            self.production_focus['Food'] = 1.5
            self.production_focus['Tech'] = 1.3
            self.production_focus['Medicine'] = 1.3
        elif self.country == 'Australia':
            # Raw Resources and Food focus
            self.production_focus['Raw Resource'] = 1.8
            self.production_focus['Food'] = 1.5
            self.production_focus['Energies'] = 1.3
        elif self.country == 'Canada':
            # Raw Resources and Energies focus
            self.production_focus['Raw Resource'] = 1.8
            self.production_focus['Energies'] = 1.5
            self.production_focus['Food'] = 1.3
        elif self.country == 'France':
            # Tech and Delicacies focus
            self.production_focus['Tech'] = 1.5
            self.production_focus['Delicacies'] = 1.8
            self.production_focus['Armaments'] = 1.3
        elif self.country == 'UK':
            # Tech and Finance focus (represented by Delicacies)
            self.production_focus['Tech'] = 1.5
            self.production_focus['Delicacies'] = 1.5
            self.production_focus['Medicine'] = 1.3
        elif self.country == 'South Korea':
            # Tech focus
            self.production_focus['Tech'] = 1.8
            self.production_focus['Refined Resources'] = 1.5
        elif self.country == 'Italy':
            # Delicacies and Refined Resources focus
            self.production_focus['Delicacies'] = 1.8
            self.production_focus['Refined Resources'] = 1.3
        elif self.country == 'Spain':
            # Food and Delicacies focus
            self.production_focus['Food'] = 1.5
            self.production_focus['Delicacies'] = 1.5
        elif self.country == 'Mexico':
            # Food and Raw Resources focus
            self.production_focus['Food'] = 1.5
            self.production_focus['Raw Resource'] = 1.3
        elif self.country == 'Indonesia':
            # Food and Raw Resources focus
            self.production_focus['Food'] = 1.5
            self.production_focus['Raw Resource'] = 1.3
        elif self.country == 'Netherlands':
            # Tech and Refined Resources focus
            self.production_focus['Tech'] = 1.5
            self.production_focus['Refined Resources'] = 1.5
        elif self.country == 'Turkey':
            # Food and Raw Resources focus
            self.production_focus['Food'] = 1.3
            self.production_focus['Raw Resource'] = 1.3
        elif self.country == 'Switzerland':
            # Tech and Medicine focus
            self.production_focus['Tech'] = 1.8
            self.production_focus['Medicine'] = 1.8
            self.production_focus['Delicacies'] = 1.5
    
    def _calculate_production_capabilities(self) -> Dict[str, float]:
        """Calculate production capabilities based on country characteristics and production focus"""
        capabilities = {}
        
        # Base capabilities (from original logic)
        base_capabilities = {}
        
        # Food production (higher for agricultural countries)
        if self.region in ['Asia', 'South America']:
            base_capabilities['Food'] = 1.5
        else:
            base_capabilities['Food'] = 1.0
            
        # Raw Resources (higher for resource-rich countries)
        if self.country in ['Russia', 'Brazil', 'Australia', 'Canada']:
            base_capabilities['Raw Resource'] = 2.0
        else:
            base_capabilities['Raw Resource'] = 1.0
            
        # Tech production (higher for developed countries)
        if self.tech_level >= 8:
            base_capabilities['Tech'] = 1.5
        else:
            base_capabilities['Tech'] = 0.5
            
        # Armaments (higher for military-industrial countries)
        if self.country in ['USA', 'Russia', 'China', 'Germany']:
            base_capabilities['Armaments'] = 1.5
        else:
            base_capabilities['Armaments'] = 0.8
            
        # Refined Resources (higher for industrialized countries)
        if self.tech_level >= 7:
            base_capabilities['Refined Resources'] = 1.3
        else:
            base_capabilities['Refined Resources'] = 0.7
            
        # Energies (higher for energy-producing countries)
        if self.country in ['Saudi Arabia', 'Russia', 'USA']:
            base_capabilities['Energies'] = 2.0
        else:
            base_capabilities['Energies'] = 1.0
            
        # Delicacies (higher for diverse economies)
        if self.gdp_per_capita > 30000:
            base_capabilities['Delicacies'] = 1.4
        else:
            base_capabilities['Delicacies'] = 0.8
            
        # Medicine (higher for developed healthcare systems)
        if self.tech_level >= 8:
            base_capabilities['Medicine'] = 1.3
        else:
            base_capabilities['Medicine'] = 0.6
        
        # Apply production focus multipliers
        for item_type, base_cap in base_capabilities.items():
            focus_multiplier = self.production_focus.get(item_type, 1.0)
            capabilities[item_type] = base_cap * focus_multiplier
            
        return capabilities
    
    def _initialize_economy(self):
        """Initialize the town's economy with base inventory and prices"""
        item_types = ['Food', 'Raw Resource', 'Tech', 'Armaments', 'Refined Resources', 
                     'Energies', 'Delicacies', 'Medicine']
        
        for item_type in item_types:
            # Base inventory based on production capabilities
            base_inventory = int(1000 * self.production_capabilities.get(item_type, 1.0))
            self.inventory[item_type] = base_inventory
            
            # Base prices with regional adjustments
            base_price = self._get_base_price(item_type)
            regional_multiplier = self._get_regional_price_multiplier(item_type)
            self.prices[item_type] = base_price * regional_multiplier
            
            # Initialize demand and supply
            self.demand[item_type] = 1000
            self.supply[item_type] = base_inventory
    
    def _get_base_price(self, item_type: str) -> float:
        """Get base price for an item type"""
        base_prices = {
            'Food': 100, 'Raw Resource': 50, 'Tech': 500, 'Armaments': 300,
            'Refined Resources': 200, 'Energies': 150, 'Delicacies': 800, 'Medicine': 400
        }
        return base_prices.get(item_type, 100)
    
    def _get_regional_price_multiplier(self, item_type: str) -> float:
        """Get regional price multiplier based on supply/demand"""
        multiplier = 1.0
        
        # Adjust based on production capabilities
        production = self.production_capabilities.get(item_type, 1.0)
        if production > 1.5:
            multiplier *= 0.8  # Oversupply
        elif production < 0.8:
            multiplier *= 1.3  # Shortage
            
        # Adjust based on GDP
        if self.gdp_per_capita > 40000:
            multiplier *= 1.2  # Higher purchasing power
        elif self.gdp_per_capita < 10000:
            multiplier *= 0.8  # Lower purchasing power
            
        return multiplier
    
    def update_prices(self, market_events: List[Dict] = None):
        """Update prices based on supply/demand and market events"""
        for item_type in self.prices.keys():
            # Base price movement
            supply_ratio = self.supply[item_type] / max(self.demand[item_type], 1)
            
            # Price adjustment based on supply/demand
            if supply_ratio < 0.5:  # Shortage
                price_change = 1.2
            elif supply_ratio > 2.0:  # Oversupply
                price_change = 0.9
            else:
                price_change = 1.0 + (1.0 - supply_ratio) * 0.1
                
            # Apply market events
            if market_events:
                for event in market_events:
                    if event.get('item_type') == item_type and event.get('country') == self.country:
                        price_change *= event.get('price_multiplier', 1.0)
            
            # Update price with some randomness
            volatility = 0.1
            random_factor = 1.0 + np.random.normal(0, volatility)
            self.prices[item_type] *= price_change * random_factor
            
            # Ensure price doesn't go below minimum
            self.prices[item_type] = max(self.prices[item_type], self._get_base_price(item_type) * 0.1)
    
    def trade(self, item_type: str, quantity: int, is_buying: bool) -> float:
        """Execute a trade and return the total cost"""
        if item_type not in self.inventory:
            return 0.0
            
        if is_buying:
            # Buying from the town
            if self.inventory[item_type] >= quantity:
                self.inventory[item_type] -= quantity
                self.supply[item_type] -= quantity
                return quantity * self.prices[item_type]
            else:
                return 0.0
        else:
            # Selling to the town
            self.inventory[item_type] += quantity
            self.supply[item_type] += quantity
            return quantity * self.prices[item_type]
    
    def get_distance_to(self, other_town: 'Town') -> float:
        """Calculate distance to another town using Haversine formula"""
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other_town.latitude), math.radians(other_town.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        return c * r

    def get_reputation_with(self, other_country: str) -> float:
        """Get reputation with another country"""
        return self.reputation.get(other_country, 0.0)
    
    def update_reputation(self, other_country: str, change: float):
        """Update reputation with another country"""
        if other_country == self.country:
            return  # Can't change self-reputation
        
        current_reputation = self.reputation.get(other_country, 0.0)
        new_reputation = max(-1.0, min(1.0, current_reputation + change))
        self.reputation[other_country] = new_reputation
        
        # Update allies/enemies lists
        if new_reputation >= 0.5 and other_country not in self.allies:
            self.allies.append(other_country)
            if other_country in self.enemies:
                self.enemies.remove(other_country)
        elif new_reputation <= -0.3 and other_country not in self.enemies:
            self.enemies.append(other_country)
            if other_country in self.allies:
                self.allies.remove(other_country)
    
    def get_production_focus_multiplier(self, item_type: str) -> float:
        """Get production focus multiplier for an item type"""
        return self.production_focus.get(item_type, 1.0)
    
    def is_ally(self, other_country: str) -> bool:
        """Check if another country is an ally"""
        return other_country in self.allies
    
    def is_enemy(self, other_country: str) -> bool:
        """Check if another country is an enemy"""
        return other_country in self.enemies
    
    def get_reputation_level(self, other_country: str) -> str:
        """Get reputation level as a string"""
        reputation = self.get_reputation_with(other_country)
        if reputation >= 0.5:
            return "Ally"
        elif reputation >= 0.2:
            return "Friendly"
        elif reputation >= -0.2:
            return "Neutral"
        elif reputation >= -0.5:
            return "Unfriendly"
        else:
            return "Enemy"

class Caravan:
    """Represents a trade route/caravan in the simulation"""
    
    def __init__(self, caravan_id: str, origin: Town, destination: Town):
        self.caravan_id = caravan_id
        self.origin = origin
        self.destination = destination
        self.current_location = origin
        self.route_progress = 0.0  # 0.0 to 1.0
        
        # Cargo
        self.cargo: Dict[str, int] = {}
        self.cargo_value = 0.0
        
        # Route characteristics
        self.distance = origin.get_distance_to(destination)
        self.travel_time = self._calculate_travel_time()
        self.risk_level = self._calculate_risk_level()
        self.reputation_impact = 0.0
        
        # Status
        self.is_active = True
        self.departure_time = None
        self.arrival_time = None
        
    def _calculate_travel_time(self) -> int:
        """Calculate travel time in days based on distance and route conditions"""
        # Base speed: 100 km/day
        base_speed = 100
        
        # Adjust for route conditions
        speed_multiplier = 1.0
        
        # Longer routes are slower due to logistics
        if self.distance > 5000:
            speed_multiplier *= 0.8
        elif self.distance > 2000:
            speed_multiplier *= 0.9
            
        # Adjust for origin/destination stability
        stability_factor = (self.origin.stability + self.destination.stability) / 18.0
        speed_multiplier *= stability_factor
        
        travel_time = int(self.distance / (base_speed * speed_multiplier))
        return max(travel_time, 1)  # Minimum 1 day
    
    def _calculate_risk_level(self) -> float:
        """Calculate risk level for this route (0.0 to 1.0)"""
        base_risk = 0.1
        
        # Distance risk
        distance_risk = min(self.distance / 10000.0, 0.3)
        
        # Stability risk
        stability_risk = (10 - min(self.origin.stability, self.destination.stability)) / 10.0 * 0.3
        
        # Regional risk
        regional_risk = 0.0
        if self.origin.region != self.destination.region:
            regional_risk = 0.2
            
        # Political risk (enhanced with reputation system)
        political_risk = self._calculate_political_risk()
        
        # Reputation risk
        reputation_risk = self._calculate_reputation_risk()
        
        total_risk = base_risk + distance_risk + stability_risk + regional_risk + political_risk + reputation_risk
        return min(total_risk, 1.0)
    
    def _calculate_political_risk(self) -> float:
        """Calculate political risk based on country relations"""
        political_risk = 0.0
        
        # Check for known hostile relationships
        if self.origin.country in ['Russia', 'China'] and self.destination.country in ['USA', 'UK']:
            political_risk = 0.3
        elif self.destination.country in ['Russia', 'China'] and self.origin.country in ['USA', 'UK']:
            political_risk = 0.3
        elif self.origin.country in ['USA', 'UK'] and self.destination.country in ['Iran', 'North Korea']:
            political_risk = 0.4
        elif self.destination.country in ['USA', 'UK'] and self.origin.country in ['Iran', 'North Korea']:
            political_risk = 0.4
        
        return political_risk
    
    def _calculate_reputation_risk(self) -> float:
        """Calculate risk based on reputation between countries"""
        reputation = self.origin.get_reputation_with(self.destination.country)
        
        # Convert reputation to risk (higher reputation = lower risk)
        if reputation >= 0.5:  # Ally
            return -0.1  # Negative risk (reduces total risk)
        elif reputation >= 0.2:  # Friendly
            return 0.0
        elif reputation >= -0.2:  # Neutral
            return 0.1
        elif reputation >= -0.5:  # Unfriendly
            return 0.2
        else:  # Enemy
            return 0.3
    
    def load_cargo(self, item_type: str, quantity: int, price_per_unit: float):
        """Load cargo onto the caravan"""
        if item_type not in self.cargo:
            self.cargo[item_type] = 0
        self.cargo[item_type] += quantity
        self.cargo_value += quantity * price_per_unit
    
    def unload_cargo(self, item_type: str) -> int:
        """Unload cargo from the caravan and return quantity"""
        quantity = self.cargo.get(item_type, 0)
        if quantity > 0:
            self.cargo[item_type] = 0
            # Update cargo value (simplified)
            self.cargo_value = sum(qty * 100 for qty in self.cargo.values())
        return quantity
    
    def update_progress(self, days_elapsed: int):
        """Update caravan progress along the route"""
        if not self.is_active:
            return
            
        progress_per_day = 1.0 / self.travel_time
        self.route_progress += progress_per_day * days_elapsed
        
        if self.route_progress >= 1.0:
            self.route_progress = 1.0
            self.current_location = self.destination
            self.is_active = False
    
    def get_current_location(self) -> Town:
        """Get the current location of the caravan"""
        if self.route_progress >= 1.0:
            return self.destination
        else:
            return self.origin  # Simplified - could interpolate position 
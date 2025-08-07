import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random

class MarketEvent:
    """Represents a market event that affects prices and trade"""
    
    def __init__(self, event_type: str, item_type: str, country: str, 
                 price_multiplier: float, duration_days: int, description: str):
        self.event_type = event_type
        self.item_type = item_type
        self.country = country
        self.price_multiplier = price_multiplier
        self.duration_days = duration_days
        self.description = description
        self.start_time = datetime.now()
        self.expiry_time = self.start_time + timedelta(days=duration_days)
        self.is_active = True
    
    def is_expired(self, current_time: datetime) -> bool:
        """Check if the event has expired"""
        return current_time > self.expiry_time
    
    def get_remaining_days(self, current_time: datetime) -> int:
        """Get remaining days for the event"""
        if self.is_expired(current_time):
            return 0
        remaining = self.expiry_time - current_time
        return remaining.days

class MarketEventGenerator:
    """Generates market events based on economic conditions"""
    
    def __init__(self):
        self.event_types = {
            'supply_shock': {
                'description': 'Supply disruption affecting production',
                'price_multiplier_range': (1.3, 2.0),
                'duration_range': (7, 30),
                'probability': 0.1
            },
            'demand_surge': {
                'description': 'Sudden increase in demand',
                'price_multiplier_range': (1.2, 1.8),
                'duration_range': (5, 21),
                'probability': 0.08
            },
            'political_instability': {
                'description': 'Political unrest affecting trade',
                'price_multiplier_range': (1.4, 2.5),
                'duration_range': (14, 60),
                'probability': 0.05
            },
            'natural_disaster': {
                'description': 'Natural disaster disrupting supply chains',
                'price_multiplier_range': (1.5, 3.0),
                'duration_range': (10, 45),
                'probability': 0.03
            },
            'technological_breakthrough': {
                'description': 'Technological advancement affecting production',
                'price_multiplier_range': (0.7, 0.9),
                'duration_range': (30, 90),
                'probability': 0.02
            },
            'trade_agreement': {
                'description': 'New trade agreement reducing barriers',
                'price_multiplier_range': (0.8, 0.95),
                'duration_range': (60, 180),
                'probability': 0.04
            }
        }
        
        self.item_specific_events = {
            'Food': ['supply_shock', 'demand_surge', 'natural_disaster'],
            'Raw Resource': ['supply_shock', 'political_instability', 'natural_disaster'],
            'Tech': ['demand_surge', 'technological_breakthrough', 'political_instability'],
            'Armaments': ['demand_surge', 'political_instability'],
            'Refined Resources': ['supply_shock', 'technological_breakthrough'],
            'Energies': ['supply_shock', 'political_instability', 'natural_disaster'],
            'Delicacies': ['demand_surge', 'supply_shock'],
            'Medicine': ['demand_surge', 'supply_shock', 'technological_breakthrough']
        }
        
        self.country_risk_factors = {
            'USA': 0.1,
            'China': 0.2,
            'Germany': 0.05,
            'Japan': 0.05,
            'India': 0.3,
            'Brazil': 0.25,
            'Russia': 0.4,
            'France': 0.1,
            'UK': 0.15,
            'Canada': 0.05,
            'Australia': 0.1,
            'South Korea': 0.1,
            'Italy': 0.2,
            'Spain': 0.15,
            'Mexico': 0.25,
            'Indonesia': 0.3,
            'Netherlands': 0.05,
            'Saudi Arabia': 0.3,
            'Turkey': 0.35,
            'Switzerland': 0.05
        }
    
    def generate_random_event(self, current_time: datetime, 
                            towns: Dict[str, Dict]) -> Optional[MarketEvent]:
        """Generate a random market event"""
        
        # Check if we should generate an event
        if random.random() > 0.05:  # 5% chance per day
            return None
        
        # Select event type
        event_type = self._select_event_type()
        if not event_type:
            return None
        
        # Select item type
        item_type = self._select_item_type(event_type)
        if not item_type:
            return None
        
        # Select country
        country = self._select_country(towns, event_type)
        if not country:
            return None
        
        # Generate event parameters
        event_config = self.event_types[event_type]
        price_multiplier = random.uniform(*event_config['price_multiplier_range'])
        duration_days = random.randint(*event_config['duration_range'])
        
        # Create event description
        description = self._create_event_description(event_type, item_type, country)
        
        return MarketEvent(event_type, item_type, country, price_multiplier, 
                          duration_days, description)
    
    def _select_event_type(self) -> Optional[str]:
        """Select an event type based on probabilities"""
        event_types = list(self.event_types.keys())
        probabilities = [self.event_types[et]['probability'] for et in event_types]
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob == 0:
            return None
        
        normalized_probs = [p / total_prob for p in probabilities]
        
        # Select based on probability
        rand_val = random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(normalized_probs):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return event_types[i]
        
        return None
    
    def _select_item_type(self, event_type: str) -> Optional[str]:
        """Select an item type for the event"""
        # Get item types that can be affected by this event
        affected_items = []
        for item_type, events in self.item_specific_events.items():
            if event_type in events:
                affected_items.append(item_type)
        
        if not affected_items:
            return None
        
        return random.choice(affected_items)
    
    def _select_country(self, towns: Dict[str, Dict], event_type: str) -> Optional[str]:
        """Select a country for the event based on risk factors"""
        available_countries = list(towns.keys())
        
        if not available_countries:
            return None
        
        # Weight countries by risk factors
        country_weights = []
        for country in available_countries:
            risk_factor = self.country_risk_factors.get(country, 0.2)
            
            # Adjust weight based on event type
            if event_type == 'political_instability':
                weight = risk_factor * 2.0  # Higher weight for political events
            elif event_type == 'natural_disaster':
                weight = risk_factor * 1.5  # Higher weight for natural disasters
            else:
                weight = risk_factor
            
            country_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(country_weights)
        if total_weight == 0:
            return random.choice(available_countries)
        
        normalized_weights = [w / total_weight for w in country_weights]
        
        # Select country based on weights
        rand_val = random.random()
        cumulative_weight = 0
        
        for i, weight in enumerate(normalized_weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return available_countries[i]
        
        return random.choice(available_countries)
    
    def _create_event_description(self, event_type: str, item_type: str, country: str) -> str:
        """Create a description for the market event"""
        descriptions = {
            'supply_shock': f"Supply disruption in {country} affecting {item_type} production",
            'demand_surge': f"Sudden increase in {item_type} demand in {country}",
            'political_instability': f"Political unrest in {country} affecting {item_type} trade",
            'natural_disaster': f"Natural disaster in {country} disrupting {item_type} supply chains",
            'technological_breakthrough': f"Technological breakthrough in {country} affecting {item_type} production",
            'trade_agreement': f"New trade agreement involving {country} affecting {item_type} markets"
        }
        
        return descriptions.get(event_type, f"Market event in {country} affecting {item_type}")
    
    def generate_seasonal_events(self, current_time: datetime) -> List[MarketEvent]:
        """Generate seasonal events based on time of year"""
        events = []
        
        # Holiday season demand surge (November-December)
        if current_time.month in [11, 12]:
            if random.random() < 0.3:  # 30% chance
                events.append(MarketEvent(
                    'demand_surge', 'Delicacies', 'USA',
                    1.4, 30, "Holiday season demand surge for delicacies"
                ))
        
        # Agricultural harvest season (September-October)
        if current_time.month in [9, 10]:
            if random.random() < 0.2:  # 20% chance
                events.append(MarketEvent(
                    'supply_shock', 'Food', 'China',
                    0.8, 21, "Harvest season increasing food supply"
                ))
        
        # Energy demand in winter (December-February)
        if current_time.month in [12, 1, 2]:
            if random.random() < 0.25:  # 25% chance
                events.append(MarketEvent(
                    'demand_surge', 'Energies', 'Germany',
                    1.3, 45, "Winter energy demand surge"
                ))
        
        return events
    
    def generate_global_events(self, current_time: datetime) -> List[MarketEvent]:
        """Generate global events that affect multiple countries"""
        events = []
        
        # Global tech boom
        if random.random() < 0.02:  # 2% chance
            events.append(MarketEvent(
                'demand_surge', 'Tech', 'Global',
                1.5, 60, "Global tech boom increasing demand"
            ))
        
        # Global supply chain disruption
        if random.random() < 0.01:  # 1% chance
            events.append(MarketEvent(
                'supply_shock', 'Raw Resource', 'Global',
                1.8, 45, "Global supply chain disruption"
            ))
        
        return events

class MarketEventManager:
    """Manages market events and their effects"""
    
    def __init__(self):
        self.active_events: List[MarketEvent] = []
        self.event_generator = MarketEventGenerator()
        self.event_history: List[MarketEvent] = []
    
    def update_events(self, current_time: datetime, towns: Dict[str, Dict]):
        """Update active events and generate new ones"""
        # Remove expired events
        self.active_events = [event for event in self.active_events 
                            if not event.is_expired(current_time)]
        
        # Generate new events
        new_event = self.event_generator.generate_random_event(current_time, towns)
        if new_event:
            self.active_events.append(new_event)
        
        # Generate seasonal events
        seasonal_events = self.event_generator.generate_seasonal_events(current_time)
        self.active_events.extend(seasonal_events)
        
        # Generate global events
        global_events = self.event_generator.generate_global_events(current_time)
        self.active_events.extend(global_events)
        
        # Update event history
        self.event_history.extend(self.active_events)
    
    def get_active_events(self, country: str = None, item_type: str = None) -> List[MarketEvent]:
        """Get active events, optionally filtered by country or item type"""
        events = self.active_events
        
        if country:
            events = [e for e in events if e.country == country or e.country == 'Global']
        
        if item_type:
            events = [e for e in events if e.item_type == item_type]
        
        return events
    
    def get_price_multiplier(self, country: str, item_type: str) -> float:
        """Get the combined price multiplier for a country and item type"""
        multiplier = 1.0
        
        for event in self.active_events:
            if (event.country == country or event.country == 'Global') and event.item_type == item_type:
                multiplier *= event.price_multiplier
        
        return multiplier
    
    def get_event_summary(self) -> Dict:
        """Get a summary of active events"""
        summary = {
            'total_active_events': len(self.active_events),
            'events_by_type': {},
            'events_by_country': {},
            'events_by_item': {}
        }
        
        for event in self.active_events:
            # Count by type
            summary['events_by_type'][event.event_type] = summary['events_by_type'].get(event.event_type, 0) + 1
            
            # Count by country
            summary['events_by_country'][event.country] = summary['events_by_country'].get(event.country, 0) + 1
            
            # Count by item
            summary['events_by_item'][event.item_type] = summary['events_by_item'].get(event.item_type, 0) + 1
        
        return summary
    
    def get_event_impact_analysis(self, towns: Dict[str, Dict]) -> Dict:
        """Analyze the impact of events on the economy"""
        impact_analysis = {
            'most_affected_countries': [],
            'most_affected_items': [],
            'price_volatility': {},
            'trade_disruption': {}
        }
        
        # Calculate impact by country
        country_impacts = {}
        for country in towns.keys():
            total_impact = 0
            for event in self.active_events:
                if event.country == country or event.country == 'Global':
                    total_impact += abs(event.price_multiplier - 1.0)
            country_impacts[country] = total_impact
        
        # Get most affected countries
        sorted_countries = sorted(country_impacts.items(), key=lambda x: x[1], reverse=True)
        impact_analysis['most_affected_countries'] = sorted_countries[:5]
        
        # Calculate impact by item
        item_impacts = {}
        for event in self.active_events:
            if event.item_type not in item_impacts:
                item_impacts[event.item_type] = 0
            item_impacts[event.item_type] += abs(event.price_multiplier - 1.0)
        
        # Get most affected items
        sorted_items = sorted(item_impacts.items(), key=lambda x: x[1], reverse=True)
        impact_analysis['most_affected_items'] = sorted_items[:5]
        
        return impact_analysis 
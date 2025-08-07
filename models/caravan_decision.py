import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class TradeDecision:
    """Represents a trade decision made by the caravan"""
    origin: str
    destination: str
    item_type: str
    quantity: int
    expected_profit: float
    risk_level: float
    travel_time: int
    confidence_score: float

class CaravanDecisionModel:
    """Decision model for caravan trade routes using quant-style analysis"""
    
    def __init__(self):
        self.risk_tolerance = 0.5  # 0.0 = risk averse, 1.0 = risk seeking
        self.min_profit_margin = 0.15  # 15% minimum profit margin
        self.max_travel_time = 30  # Maximum travel time in days
        self.reputation_threshold = 0.7  # Minimum reputation for high-value trades
        
        # Decision weights
        self.weights = {
            'price_arbitrage': 0.4,
            'travel_time': 0.2,
            'risk_level': 0.2,
            'reputation_impact': 0.1,
            'market_efficiency': 0.1
        }
    
    def evaluate_trade_opportunity(self, origin_town: Dict, dest_town: Dict,
                                 item_type: str, economy_model) -> Optional[TradeDecision]:
        """Evaluate a trade opportunity and return a decision"""
        
        # Get current prices
        origin_price = origin_town.get('prices', {}).get(item_type, 0)
        dest_price = dest_town.get('prices', {}).get(item_type, 0)
        
        if origin_price <= 0 or dest_price <= 0:
            return None
        
        # Calculate basic metrics
        price_ratio = dest_price / origin_price
        price_difference = dest_price - origin_price
        profit_margin = (price_ratio - 1) * 100
        
        # Check minimum profit margin
        if profit_margin < self.min_profit_margin * 100:
            return None
        
        # Calculate travel time and distance
        travel_time = self._calculate_travel_time(origin_town, dest_town)
        if travel_time > self.max_travel_time:
            return None
        
        # Calculate risk level (enhanced with reputation)
        risk_level = self._calculate_risk_level(origin_town, dest_town, item_type)
        
        # Calculate reputation impact
        reputation_impact = self._calculate_reputation_impact(origin_town, dest_town, item_type)
        
        # Calculate production focus bonus
        production_focus_bonus = self._calculate_production_focus_bonus(origin_town, dest_town, item_type)
        
        # Calculate optimal quantity (considering production focus)
        optimal_quantity = self._calculate_optimal_quantity(
            origin_town, dest_town, item_type, price_ratio, production_focus_bonus
        )
        
        if optimal_quantity <= 0:
            return None
        
        # Calculate expected profit (enhanced with reputation and production focus)
        expected_profit = self._calculate_expected_profit(
            origin_price, dest_price, optimal_quantity, risk_level, travel_time,
            reputation_impact, production_focus_bonus
        )
        
        # Calculate confidence score (enhanced with reputation)
        confidence_score = self._calculate_confidence_score(
            origin_town, dest_town, item_type, price_ratio, risk_level, reputation_impact
        )
        
        # Make final decision
        if confidence_score > 0.6:  # Minimum confidence threshold
            return TradeDecision(
                origin=origin_town.get('country', ''),
                destination=dest_town.get('country', ''),
                item_type=item_type,
                quantity=optimal_quantity,
                expected_profit=expected_profit,
                risk_level=risk_level,
                travel_time=travel_time,
                confidence_score=confidence_score
            )
        
        return None
    
    def _calculate_travel_time(self, origin_town: Dict, dest_town: Dict) -> int:
        """Calculate travel time between towns"""
        # Simplified distance calculation
        origin_lat = origin_town.get('latitude', 0)
        origin_lon = origin_town.get('longitude', 0)
        dest_lat = dest_town.get('latitude', 0)
        dest_lon = dest_town.get('longitude', 0)
        
        # Haversine distance calculation
        lat1, lon1 = math.radians(origin_lat), math.radians(origin_lon)
        lat2, lon2 = math.radians(dest_lat), math.radians(dest_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in kilometers
        r = 6371
        distance = c * r
        
        # Base speed: 100 km/day, adjusted for route conditions
        base_speed = 100
        
        # Adjust for stability
        stability_factor = (origin_town.get('stability', 5) + dest_town.get('stability', 5)) / 18.0
        base_speed *= stability_factor
        
        travel_time = int(distance / base_speed)
        return max(travel_time, 1)
    
    def _calculate_risk_level(self, origin_town: Dict, dest_town: Dict, item_type: str) -> float:
        """Calculate risk level for the trade route (0.0 to 1.0)"""
        base_risk = 0.1
        
        # Distance risk
        distance = self._calculate_distance(origin_town, dest_town)
        distance_risk = min(distance / 10000.0, 0.3)
        
        # Stability risk
        origin_stability = origin_town.get('stability', 5)
        dest_stability = dest_town.get('stability', 5)
        stability_risk = (10 - min(origin_stability, dest_stability)) / 10.0 * 0.3
        
        # Political risk (enhanced with reputation)
        political_risk = self._calculate_political_risk(origin_town, dest_town)
        
        # Reputation risk
        reputation_risk = self._calculate_reputation_risk(origin_town, dest_town)
        
        # Item-specific risk
        item_risk = self._calculate_item_risk(item_type)
        
        # Regional risk
        regional_risk = 0.0
        if origin_town.get('region') != dest_town.get('region'):
            regional_risk = 0.2
        
        total_risk = base_risk + distance_risk + stability_risk + political_risk + reputation_risk + item_risk + regional_risk
        return min(total_risk, 1.0)
    
    def _calculate_reputation_risk(self, origin_town: Dict, dest_town: Dict) -> float:
        """Calculate risk based on reputation between countries"""
        origin_country = origin_town.get('country', '')
        dest_country = dest_town.get('country', '')
        
        reputation = self._get_reputation_between_countries(origin_country, dest_country)
        
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
    
    def _calculate_distance(self, origin_town: Dict, dest_town: Dict) -> float:
        """Calculate distance between towns"""
        origin_lat = origin_town.get('latitude', 0)
        origin_lon = origin_town.get('longitude', 0)
        dest_lat = dest_town.get('latitude', 0)
        dest_lon = dest_town.get('longitude', 0)
        
        lat1, lon1 = math.radians(origin_lat), math.radians(origin_lon)
        lat2, lon2 = math.radians(dest_lat), math.radians(dest_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        r = 6371
        return c * r
    
    def _calculate_political_risk(self, origin_town: Dict, dest_town: Dict) -> float:
        """Calculate political risk between countries"""
        origin_country = origin_town.get('country', '')
        dest_country = dest_town.get('country', '')
        
        # Simplified political risk calculation
        high_risk_pairs = [
            ('USA', 'Russia'), ('USA', 'China'), ('Russia', 'China'),
            ('UK', 'Russia'), ('Germany', 'Russia')
        ]
        
        pair = tuple(sorted([origin_country, dest_country]))
        if pair in high_risk_pairs:
            return 0.3
        
        return 0.0
    
    def _calculate_item_risk(self, item_type: str) -> float:
        """Calculate item-specific risk"""
        risk_levels = {
            'Food': 0.1,  # Low risk
            'Raw Resource': 0.2,  # Medium risk
            'Tech': 0.3,  # High risk (sanctions, etc.)
            'Armaments': 0.4,  # Very high risk
            'Refined Resources': 0.2,
            'Energies': 0.3,
            'Delicacies': 0.1,
            'Medicine': 0.2
        }
        
        return risk_levels.get(item_type, 0.2)
    
    def _calculate_optimal_quantity(self, origin_town: Dict, dest_town: Dict,
                                  item_type: str, price_ratio: float, production_focus_bonus: float) -> int:
        """Calculate optimal trade quantity"""
        origin_supply = origin_town.get('inventory', {}).get(item_type, 0)
        dest_demand = dest_town.get('demand', {}).get(item_type, 1000)
        
        # Base quantity
        base_quantity = min(origin_supply, dest_demand, 1000)  # Max capacity
        
        # Adjust based on price ratio
        if price_ratio > 1.5:  # High profit margin
            base_quantity = int(base_quantity * 1.2)
        elif price_ratio < 1.2:  # Low profit margin
            base_quantity = int(base_quantity * 0.8)
        
        # Adjust based on risk
        risk_level = self._calculate_risk_level(origin_town, dest_town, item_type)
        if risk_level > 0.7:
            base_quantity = int(base_quantity * 0.5)  # Reduce quantity for high risk
        
        # Apply production focus bonus
        optimal_quantity = int(base_quantity * (1.0 + production_focus_bonus))
        
        return max(optimal_quantity, 1)
    
    def _calculate_expected_profit(self, origin_price: float, dest_price: float,
                                 quantity: int, risk_level: float, travel_time: int,
                                 reputation_impact: float, production_focus_bonus: float) -> float:
        """Calculate expected profit considering risk, time, reputation, and production focus"""
        gross_profit = (dest_price - origin_price) * quantity
        
        # Apply risk adjustment
        risk_adjustment = 1.0 - risk_level
        adjusted_profit = gross_profit * risk_adjustment
        
        # Apply time discount
        time_discount = 1.0 / (1.0 + travel_time * 0.01)  # 1% per day
        expected_profit = adjusted_profit * time_discount
        
        # Apply reputation impact
        expected_profit *= (1.0 + reputation_impact)
        
        # Apply production focus bonus
        expected_profit *= (1.0 + production_focus_bonus)
        
        return expected_profit
    
    def _calculate_confidence_score(self, origin_town: Dict, dest_town: Dict,
                                  item_type: str, price_ratio: float, risk_level: float, reputation_impact: float) -> float:
        """Calculate confidence score for the trade decision"""
        confidence = 0.5  # Base confidence
        
        # Price ratio confidence
        if price_ratio > 1.5:
            confidence += 0.2
        elif price_ratio > 1.2:
            confidence += 0.1
        
        # Stability confidence
        origin_stability = origin_town.get('stability', 5)
        dest_stability = dest_town.get('stability', 5)
        avg_stability = (origin_stability + dest_stability) / 2.0
        stability_confidence = avg_stability / 10.0
        confidence += stability_confidence * 0.2
        
        # Risk confidence
        risk_confidence = 1.0 - risk_level
        confidence += risk_confidence * 0.2
        
        # Reputation confidence
        confidence += reputation_impact * 0.2
        
        # Market efficiency confidence
        market_efficiency = self._calculate_market_efficiency(origin_town, dest_town, item_type)
        confidence += market_efficiency * 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_market_efficiency(self, origin_town: Dict, dest_town: Dict, item_type: str) -> float:
        """Calculate market efficiency score"""
        origin_price = origin_town.get('prices', {}).get(item_type, 0)
        dest_price = dest_town.get('prices', {}).get(item_type, 0)
        
        if origin_price <= 0 or dest_price <= 0:
            return 0.0
        
        # Calculate price dispersion
        mean_price = (origin_price + dest_price) / 2.0
        price_dispersion = abs(dest_price - origin_price) / mean_price
        
        # Convert to efficiency score (lower dispersion = higher efficiency)
        efficiency = max(0.0, 1.0 - price_dispersion)
        return efficiency
    
    def _calculate_reputation_impact(self, origin_town: Dict, dest_town: Dict, item_type: str) -> float:
        """Calculate reputation impact on trade success and pricing"""
        origin_country = origin_town.get('country', '')
        dest_country = dest_town.get('country', '')
        
        # Get reputation between countries (assuming we have access to Town objects)
        # For now, we'll use a simplified approach based on known relationships
        reputation = self._get_reputation_between_countries(origin_country, dest_country)
        
        # Reputation impact on pricing and success rate
        if reputation >= 0.5:  # Ally
            return 0.2  # 20% bonus for allies
        elif reputation >= 0.2:  # Friendly
            return 0.1  # 10% bonus for friendly nations
        elif reputation >= -0.2:  # Neutral
            return 0.0  # No impact for neutral nations
        elif reputation >= -0.5:  # Unfriendly
            return -0.1  # 10% penalty for unfriendly nations
        else:  # Enemy
            return -0.3  # 30% penalty for enemies
    
    def _calculate_production_focus_bonus(self, origin_town: Dict, dest_town: Dict, item_type: str) -> float:
        """Calculate production focus bonus for the trade"""
        origin_country = origin_town.get('country', '')
        dest_country = dest_town.get('country', '')
        
        # Get production focus for both countries
        origin_focus = self._get_production_focus(origin_country, item_type)
        dest_focus = self._get_production_focus(dest_country, item_type)
        
        # Calculate bonus based on production focus
        # Higher focus means better production efficiency and potentially better prices
        origin_bonus = (origin_focus - 1.0) * 0.1  # 10% of focus multiplier
        dest_bonus = (dest_focus - 1.0) * 0.05  # 5% of focus multiplier for destination
        
        return origin_bonus + dest_bonus
    
    def _get_reputation_between_countries(self, country1: str, country2: str) -> float:
        """Get reputation between two countries (simplified implementation)"""
        # Simplified reputation system based on known relationships
        alliances = {
            ('USA', 'UK'): 0.8, ('USA', 'Canada'): 0.9, ('USA', 'Germany'): 0.7,
            ('USA', 'France'): 0.7, ('USA', 'Japan'): 0.8, ('USA', 'South Korea'): 0.8,
            ('UK', 'Canada'): 0.8, ('UK', 'Australia'): 0.8, ('UK', 'Germany'): 0.7,
            ('Germany', 'France'): 0.8, ('Germany', 'Italy'): 0.7, ('Germany', 'Spain'): 0.6,
            ('China', 'Russia'): 0.6, ('Japan', 'South Korea'): 0.5, ('Japan', 'Australia'): 0.6,
            ('Canada', 'Australia'): 0.7, ('France', 'Italy'): 0.7, ('France', 'Spain'): 0.6,
            ('Italy', 'Spain'): 0.7, ('Netherlands', 'Germany'): 0.8, ('Switzerland', 'Germany'): 0.8
        }
        
        enemies = {
            ('USA', 'Russia'): -0.6, ('USA', 'China'): -0.5, ('USA', 'Iran'): -0.8,
            ('UK', 'Russia'): -0.6, ('Germany', 'Russia'): -0.5, ('France', 'Russia'): -0.5,
            ('China', 'Japan'): -0.4, ('China', 'India'): -0.3, ('Russia', 'Ukraine'): -0.8,
            ('India', 'Pakistan'): -0.7, ('Turkey', 'Syria'): -0.6, ('Saudi Arabia', 'Iran'): -0.7
        }
        
        # Check for exact matches
        pair = tuple(sorted([country1, country2]))
        
        if pair in alliances:
            return alliances[pair]
        elif pair in enemies:
            return enemies[pair]
        else:
            # Default to neutral
            return 0.0
    
    def _get_production_focus(self, country: str, item_type: str) -> float:
        """Get production focus multiplier for a country and item type"""
        # Production focus data (simplified)
        focus_data = {
            'USA': {
                'Tech': 2.0, 'Armaments': 2.0, 'Medicine': 1.5,
                'Food': 1.0, 'Raw Resource': 1.0, 'Refined Resources': 1.0,
                'Energies': 1.0, 'Delicacies': 1.0
            },
            'China': {
                'Tech': 1.8, 'Refined Resources': 1.8, 'Raw Resource': 1.5,
                'Food': 1.0, 'Armaments': 1.0, 'Energies': 1.0,
                'Delicacies': 1.0, 'Medicine': 1.0
            },
            'Germany': {
                'Tech': 1.8, 'Refined Resources': 1.8, 'Armaments': 1.5,
                'Food': 1.0, 'Raw Resource': 1.0, 'Energies': 1.0,
                'Delicacies': 1.0, 'Medicine': 1.0
            },
            'Japan': {
                'Tech': 2.0, 'Delicacies': 1.8, 'Medicine': 1.5,
                'Food': 1.0, 'Raw Resource': 1.0, 'Armaments': 1.0,
                'Refined Resources': 1.0, 'Energies': 1.0
            },
            'Russia': {
                'Raw Resource': 2.0, 'Energies': 2.0, 'Armaments': 1.5,
                'Food': 1.0, 'Tech': 1.0, 'Refined Resources': 1.0,
                'Delicacies': 1.0, 'Medicine': 1.0
            },
            'Saudi Arabia': {
                'Energies': 2.5, 'Raw Resource': 1.5,
                'Food': 1.0, 'Tech': 1.0, 'Armaments': 1.0,
                'Refined Resources': 1.0, 'Delicacies': 1.0, 'Medicine': 1.0
            },
            'Brazil': {
                'Food': 1.8, 'Raw Resource': 1.8, 'Delicacies': 1.5,
                'Tech': 1.0, 'Armaments': 1.0, 'Refined Resources': 1.0,
                'Energies': 1.0, 'Medicine': 1.0
            },
            'India': {
                'Food': 1.5, 'Tech': 1.3, 'Medicine': 1.3,
                'Raw Resource': 1.0, 'Armaments': 1.0, 'Refined Resources': 1.0,
                'Energies': 1.0, 'Delicacies': 1.0
            },
            'Australia': {
                'Raw Resource': 1.8, 'Food': 1.5, 'Energies': 1.3,
                'Tech': 1.0, 'Armaments': 1.0, 'Refined Resources': 1.0,
                'Delicacies': 1.0, 'Medicine': 1.0
            },
            'Canada': {
                'Raw Resource': 1.8, 'Energies': 1.5, 'Food': 1.3,
                'Tech': 1.0, 'Armaments': 1.0, 'Refined Resources': 1.0,
                'Delicacies': 1.0, 'Medicine': 1.0
            }
        }
        
        # Get focus for the country and item type
        country_focus = focus_data.get(country, {})
        return country_focus.get(item_type, 1.0)
    
    def rank_trade_opportunities(self, opportunities: List[TradeDecision]) -> List[TradeDecision]:
        """Rank trade opportunities by overall score"""
        for opportunity in opportunities:
            # Calculate overall score
            score = (
                self.weights['price_arbitrage'] * (opportunity.expected_profit / 1000) +
                self.weights['travel_time'] * (1.0 - opportunity.travel_time / self.max_travel_time) +
                self.weights['risk_level'] * (1.0 - opportunity.risk_level) +
                self.weights['reputation_impact'] * opportunity.confidence_score +
                self.weights['market_efficiency'] * self._calculate_market_efficiency(
                    {'country': opportunity.origin}, 
                    {'country': opportunity.destination}, 
                    opportunity.item_type
                )
            )
            opportunity.overall_score = score
        
        # Sort by overall score
        opportunities.sort(key=lambda x: x.overall_score, reverse=True)
        return opportunities
    
    def make_decision(self, available_opportunities: List[Dict], 
                     current_reputation: float = 0.5) -> Optional[TradeDecision]:
        """Make the final trade decision"""
        if not available_opportunities:
            return None
        
        # Filter opportunities based on reputation
        filtered_opportunities = []
        for opp in available_opportunities:
            if opp['risk_level'] > 0.8 and current_reputation < self.reputation_threshold:
                continue  # Skip high-risk trades if reputation is low
            filtered_opportunities.append(opp)
        
        if not filtered_opportunities:
            return None
        
        # Rank opportunities
        ranked_opportunities = self.rank_trade_opportunities(filtered_opportunities)
        
        # Select best opportunity
        best_opportunity = ranked_opportunities[0]
        
        # Apply risk tolerance
        if self.risk_tolerance < 0.3 and best_opportunity.risk_level > 0.6:
            # Risk-averse: look for lower-risk alternatives
            for opp in ranked_opportunities[1:]:
                if opp.risk_level < 0.4:
                    return opp
        
        return best_opportunity
    
    def update_risk_tolerance(self, recent_performance: List[float]):
        """Update risk tolerance based on recent performance"""
        if not recent_performance:
            return
        
        avg_performance = np.mean(recent_performance)
        
        # Adjust risk tolerance based on performance
        if avg_performance > 0.2:  # Good performance
            self.risk_tolerance = min(1.0, self.risk_tolerance + 0.1)
        elif avg_performance < -0.1:  # Poor performance
            self.risk_tolerance = max(0.0, self.risk_tolerance - 0.1)
    
    def get_decision_summary(self, decision: TradeDecision) -> Dict:
        """Get a summary of the trade decision"""
        return {
            'route': f"{decision.origin} -> {decision.destination}",
            'item_type': decision.item_type,
            'quantity': decision.quantity,
            'expected_profit': f"${decision.expected_profit:.2f}",
            'risk_level': f"{decision.risk_level:.1%}",
            'travel_time': f"{decision.travel_time} days",
            'confidence': f"{decision.confidence_score:.1%}",
            'profit_margin': f"{((decision.expected_profit / (decision.quantity * 100)) - 1) * 100:.1f}%"
        } 
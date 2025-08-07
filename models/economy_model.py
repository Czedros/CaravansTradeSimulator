import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class EconomyModel:
    """Quant-style economy model using regression and predictive modeling"""
    
    def __init__(self):
        self.price_models: Dict[str, object] = {}
        self.demand_models: Dict[str, object] = {}
        self.supply_models: Dict[str, object] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Model parameters
        self.price_history_length = 30  # Days of price history to consider
        self.feature_columns = [
            'price_lag_1', 'price_lag_7', 'price_lag_30',
            'demand_lag_1', 'supply_lag_1', 'gdp_per_capita',
            'tech_level', 'stability', 'population',
            'distance_to_major_markets', 'regional_demand',
            'global_price_index', 'volatility_index'
        ]
        
        # Model types
        self.model_types = {
            'price': RandomForestRegressor(n_estimators=100, random_state=42),
            'demand': Ridge(alpha=1.0),
            'supply': LinearRegression()
        }
    
    def prepare_features(self, town_data: Dict, item_type: str, 
                        price_history: List[float], global_data: Dict) -> np.ndarray:
        """Prepare features for the model"""
        features = []
        
        # Price lags
        if len(price_history) >= 1:
            features.append(price_history[-1])
        else:
            features.append(0.0)
            
        if len(price_history) >= 7:
            features.append(price_history[-7])
        else:
            features.append(0.0)
            
        if len(price_history) >= 30:
            features.append(price_history[-30])
        else:
            features.append(0.0)
        
        # Demand and supply lags (simplified)
        features.extend([town_data.get('demand', {}).get(item_type, 1000), 
                        town_data.get('supply', {}).get(item_type, 1000)])
        
        # Economic indicators
        features.extend([
            town_data.get('gdp_per_capita', 0),
            town_data.get('tech_level', 5),
            town_data.get('stability', 5),
            town_data.get('population', 1000000)
        ])
        
        # Distance to major markets (simplified)
        major_markets = ['USA', 'China', 'Germany', 'Japan']
        distance_to_major = min([
            self._calculate_distance(town_data.get('country', ''), market)
            for market in major_markets
        ])
        features.append(distance_to_major)
        
        # Regional demand (simplified)
        regional_demand = global_data.get('regional_demand', {}).get(item_type, 1000)
        features.append(regional_demand)
        
        # Global price index
        global_price = global_data.get('global_price_index', {}).get(item_type, 100)
        features.append(global_price)
        
        # Volatility index
        if len(price_history) > 1:
            volatility = np.std(price_history[-30:]) if len(price_history) >= 30 else np.std(price_history)
        else:
            volatility = 0.0
        features.append(volatility)
        
        return np.array(features).reshape(1, -1)
    
    def _calculate_distance(self, country1: str, country2: str) -> float:
        """Calculate simplified distance between countries"""
        # Simplified distance calculation
        distances = {
            ('USA', 'China'): 11000, ('USA', 'Germany'): 7000, ('USA', 'Japan'): 10000,
            ('China', 'Germany'): 8000, ('China', 'Japan'): 3000, ('Germany', 'Japan'): 9000
        }
        
        key = tuple(sorted([country1, country2]))
        return distances.get(key, 5000)  # Default distance
    
    def train_price_model(self, item_type: str, training_data: List[Dict]):
        """Train price prediction model for a specific item type"""
        if not training_data:
            return
        
        X = []
        y = []
        
        for data_point in training_data:
            features = self.prepare_features(
                data_point['town_data'], 
                item_type, 
                data_point['price_history'],
                data_point['global_data']
            )
            X.append(features.flatten())
            y.append(data_point['current_price'])
        
        if len(X) < 10:  # Need minimum data points
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = self.model_types['price']
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and scaler
        self.price_models[item_type] = model
        self.scalers[f'price_{item_type}'] = scaler
        
        print(f"Trained price model for {item_type}: MSE={mse:.2f}, R²={r2:.2f}")
    
    def predict_price(self, item_type: str, town_data: Dict, 
                     price_history: List[float], global_data: Dict) -> float:
        """Predict price for an item type in a specific town"""
        if item_type not in self.price_models:
            # Return simple prediction based on historical average
            if price_history:
                base_prediction = np.mean(price_history[-10:]) if len(price_history) >= 10 else price_history[-1]
            else:
                base_prediction = town_data.get('prices', {}).get(item_type, 100)
        else:
            # Prepare features
            features = self.prepare_features(town_data, item_type, price_history, global_data)
            
            # Scale features
            scaler_key = f'price_{item_type}'
            if scaler_key in self.scalers:
                features_scaled = self.scalers[scaler_key].transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            base_prediction = self.price_models[item_type].predict(features_scaled)[0]
        
        # Apply reputation and production focus adjustments
        adjusted_prediction = self._apply_reputation_and_focus_adjustments(
            base_prediction, town_data, item_type
        )
        
        # Ensure prediction is reasonable
        base_price = town_data.get('prices', {}).get(item_type, 100)
        adjusted_prediction = max(adjusted_prediction, base_price * 0.1)  # Minimum 10% of base price
        adjusted_prediction = min(adjusted_prediction, base_price * 10.0)  # Maximum 10x base price
        
        return adjusted_prediction
    
    def _apply_reputation_and_focus_adjustments(self, base_price: float, town_data: Dict, item_type: str) -> float:
        """Apply reputation and production focus adjustments to price"""
        adjusted_price = base_price
        
        # Production focus adjustment
        production_focus = town_data.get('production_focus', {}).get(item_type, 1.0)
        if production_focus > 1.0:
            # Higher production focus means more supply, potentially lower prices
            supply_adjustment = 1.0 / (1.0 + (production_focus - 1.0) * 0.1)
            adjusted_price *= supply_adjustment
        
        # Reputation adjustment (simplified - in practice, this would depend on trading partners)
        # For now, we'll use a neutral adjustment
        reputation_adjustment = 1.0
        adjusted_price *= reputation_adjustment
        
        return adjusted_price
    
    def calculate_arbitrage_opportunities(self, towns: Dict[str, Dict], 
                                       item_type: str) -> List[Dict]:
        """Calculate arbitrage opportunities across towns"""
        opportunities = []
        
        for origin_country, origin_data in towns.items():
            for dest_country, dest_data in towns.items():
                if origin_country == dest_country:
                    continue
                
                origin_price = origin_data.get('prices', {}).get(item_type, 0)
                dest_price = dest_data.get('prices', {}).get(item_type, 0)
                
                if origin_price > 0 and dest_price > 0:
                    price_difference = dest_price - origin_price
                    price_ratio = dest_price / origin_price
                    
                    # Calculate reputation impact
                    reputation_impact = self._calculate_reputation_impact_for_arbitrage(
                        origin_country, dest_country
                    )
                    
                    # Calculate production focus impact
                    production_focus_impact = self._calculate_production_focus_impact_for_arbitrage(
                        origin_data, dest_data, item_type
                    )
                    
                    # Adjust price ratio based on reputation and production focus
                    adjusted_price_ratio = price_ratio * (1.0 + reputation_impact + production_focus_impact)
                    
                    # Only consider opportunities with significant adjusted price difference
                    if adjusted_price_ratio > 1.15:  # 15% adjusted price difference
                        opportunity = {
                            'origin': origin_country,
                            'destination': dest_country,
                            'item_type': item_type,
                            'origin_price': origin_price,
                            'destination_price': dest_price,
                            'price_difference': price_difference,
                            'price_ratio': price_ratio,
                            'adjusted_price_ratio': adjusted_price_ratio,
                            'reputation_impact': reputation_impact,
                            'production_focus_impact': production_focus_impact,
                            'potential_profit_margin': (adjusted_price_ratio - 1) * 100
                        }
                        opportunities.append(opportunity)
        
        # Sort by adjusted potential profit margin
        opportunities.sort(key=lambda x: x['potential_profit_margin'], reverse=True)
        return opportunities
    
    def _calculate_reputation_impact_for_arbitrage(self, origin_country: str, dest_country: str) -> float:
        """Calculate reputation impact on arbitrage opportunity"""
        # Simplified reputation system for arbitrage
        alliances = {
            ('USA', 'UK'): 0.1, ('USA', 'Canada'): 0.15, ('USA', 'Germany'): 0.1,
            ('USA', 'France'): 0.1, ('USA', 'Japan'): 0.1, ('USA', 'South Korea'): 0.1,
            ('UK', 'Canada'): 0.1, ('UK', 'Australia'): 0.1, ('UK', 'Germany'): 0.1,
            ('Germany', 'France'): 0.1, ('Germany', 'Italy'): 0.05, ('Germany', 'Spain'): 0.05,
            ('China', 'Russia'): 0.05, ('Japan', 'South Korea'): 0.05, ('Japan', 'Australia'): 0.05,
            ('Canada', 'Australia'): 0.1, ('France', 'Italy'): 0.05, ('France', 'Spain'): 0.05,
            ('Italy', 'Spain'): 0.05, ('Netherlands', 'Germany'): 0.1, ('Switzerland', 'Germany'): 0.1
        }
        
        enemies = {
            ('USA', 'Russia'): -0.2, ('USA', 'China'): -0.15, ('USA', 'Iran'): -0.3,
            ('UK', 'Russia'): -0.2, ('Germany', 'Russia'): -0.15, ('France', 'Russia'): -0.15,
            ('China', 'Japan'): -0.1, ('China', 'India'): -0.1, ('Russia', 'Ukraine'): -0.3,
            ('India', 'Pakistan'): -0.25, ('Turkey', 'Syria'): -0.2, ('Saudi Arabia', 'Iran'): -0.25
        }
        
        # Check for exact matches
        pair = tuple(sorted([origin_country, dest_country]))
        
        if pair in alliances:
            return alliances[pair]
        elif pair in enemies:
            return enemies[pair]
        else:
            # Default to neutral
            return 0.0
    
    def _calculate_production_focus_impact_for_arbitrage(self, origin_data: Dict, dest_data: Dict, item_type: str) -> float:
        """Calculate production focus impact on arbitrage opportunity"""
        origin_focus = origin_data.get('production_focus', {}).get(item_type, 1.0)
        dest_focus = dest_data.get('production_focus', {}).get(item_type, 1.0)
        
        # Higher origin focus means more supply (potentially lower prices)
        # Higher dest focus means more demand (potentially higher prices)
        origin_impact = (origin_focus - 1.0) * 0.05  # 5% per focus level
        dest_impact = (dest_focus - 1.0) * 0.03  # 3% per focus level
        
        return dest_impact - origin_impact  # Positive if destination has higher focus
    
    def calculate_optimal_trade_quantity(self, origin_town: Dict, dest_town: Dict,
                                       item_type: str, max_capacity: int = 1000) -> int:
        """Calculate optimal trade quantity based on supply/demand, prices, and production focus"""
        origin_supply = origin_town.get('inventory', {}).get(item_type, 0)
        dest_demand = dest_town.get('demand', {}).get(item_type, 1000)
        
        # Base quantity on supply and demand
        base_quantity = min(origin_supply, dest_demand, max_capacity)
        
        # Adjust based on price difference
        origin_price = origin_town.get('prices', {}).get(item_type, 0)
        dest_price = dest_town.get('prices', {}).get(item_type, 0)
        
        if origin_price > 0 and dest_price > 0:
            price_ratio = dest_price / origin_price
            if price_ratio > 1.5:  # High profit margin
                base_quantity = int(base_quantity * 1.2)  # Increase quantity
            elif price_ratio < 1.1:  # Low profit margin
                base_quantity = int(base_quantity * 0.8)  # Decrease quantity
        
        # Adjust based on production focus
        origin_focus = origin_town.get('production_focus', {}).get(item_type, 1.0)
        dest_focus = dest_town.get('production_focus', {}).get(item_type, 1.0)
        
        # Higher origin focus means more efficient production (more supply available)
        if origin_focus > 1.0:
            focus_multiplier = 1.0 + (origin_focus - 1.0) * 0.2  # 20% bonus per focus level
            base_quantity = int(base_quantity * focus_multiplier)
        
        # Higher dest focus means more demand (but also potentially more local supply)
        if dest_focus > 1.0:
            # Reduce quantity slightly as destination may have more local supply
            focus_reduction = 1.0 - (dest_focus - 1.0) * 0.1  # 10% reduction per focus level
            base_quantity = int(base_quantity * focus_reduction)
        
        return max(base_quantity, 1)  # Minimum 1 unit
    
    def forecast_price_trends(self, item_type: str, town_data: Dict,
                            price_history: List[float], days_ahead: int = 7) -> List[float]:
        """Forecast price trends for the next N days"""
        if not price_history or len(price_history) < 5:
            return [town_data.get('prices', {}).get(item_type, 100)] * days_ahead
        
        # Simple trend-based forecasting
        recent_prices = price_history[-10:] if len(price_history) >= 10 else price_history
        
        # Calculate trend
        if len(recent_prices) > 1:
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        else:
            trend = 0.0
        
        # Generate forecast
        forecast = []
        current_price = recent_prices[-1]
        
        for day in range(days_ahead):
            # Apply trend with some randomness
            random_factor = 1.0 + np.random.normal(0, 0.02)  # 2% daily volatility
            forecast_price = current_price * (1 + trend * 0.1) * random_factor
            
            # Ensure reasonable bounds
            base_price = town_data.get('prices', {}).get(item_type, 100)
            forecast_price = max(forecast_price, base_price * 0.1)
            forecast_price = min(forecast_price, base_price * 10.0)
            
            forecast.append(forecast_price)
            current_price = forecast_price
        
        return forecast
    
    def calculate_risk_adjusted_return(self, trade_opportunity: Dict, 
                                     risk_factors: Dict) -> float:
        """Calculate risk-adjusted return for a trade opportunity"""
        base_return = trade_opportunity['potential_profit_margin']
        
        # Risk factors
        distance_risk = risk_factors.get('distance_risk', 0.1)
        stability_risk = risk_factors.get('stability_risk', 0.1)
        political_risk = risk_factors.get('political_risk', 0.1)
        market_volatility = risk_factors.get('market_volatility', 0.1)
        
        # Calculate risk-adjusted return
        total_risk = distance_risk + stability_risk + political_risk + market_volatility
        risk_adjusted_return = base_return * (1 - total_risk)
        
        return max(risk_adjusted_return, 0.0)
    
    def get_market_efficiency_score(self, towns: Dict[str, Dict], item_type: str) -> float:
        """Calculate market efficiency score (0-1) based on price dispersion"""
        prices = []
        for town_data in towns.values():
            price = town_data.get('prices', {}).get(item_type, 0)
            if price > 0:
                prices.append(price)
        
        if len(prices) < 2:
            return 1.0  # Perfect efficiency if only one price
        
        # Calculate coefficient of variation
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        cv = std_price / mean_price if mean_price > 0 else 0
        
        # Convert to efficiency score (lower CV = higher efficiency)
        efficiency_score = max(0.0, 1.0 - cv)
        return efficiency_score 
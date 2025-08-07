import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import json

from .world_state import WorldState
from .entities import Town, Caravan, Item
from models.economy_model import EconomyModel
from models.caravan_decision import CaravanDecisionModel, TradeDecision
from models.market_events import MarketEventManager

class EconomicSimulator:
    """Main economic simulator that orchestrates the entire simulation"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        
        # Initialize core components
        self.world_state = WorldState()
        self.economy_model = EconomyModel()
        self.decision_model = CaravanDecisionModel()
        self.market_events = MarketEventManager()
        
        # Simulation state
        self.current_day = 0
        self.simulation_speed = self.config.get('simulation_speed', 1)
        self.max_days = self.config.get('max_days', 365)
        self.is_running = False
        
        # Statistics and logging
        self.daily_stats = []
        self.trade_history = []
        self.price_history = {}
        
        # Performance tracking
        self.start_time = None
        self.total_trades = 0
        self.total_profit = 0.0
        
    def _get_default_config(self) -> Dict:
        """Get default simulation configuration"""
        return {
            'simulation_speed': 1,  # Days per second
            'max_days': 365,
            'max_caravans': 50,
            'min_profit_margin': 0.15,
            'risk_tolerance': 0.5,
            'enable_market_events': True,
            'enable_visualization': True,
            'save_trade_logs': True
        }
    
    def start_simulation(self):
        """Start the economic simulation"""
        print("Starting Economic Simulator...")
        self.is_running = True
        self.start_time = datetime.now()
        
        # Initialize price history
        for item_type in self.world_state.items.keys():
            self.price_history[item_type] = []
        
        print(f"Simulation started with {len(self.world_state.towns)} towns and {len(self.world_state.items)} item types")
    
    def run_simulation_step(self, days: int = 1):
        """Run a single simulation step"""
        if not self.is_running:
            return
        
        for day in range(days):
            self.current_day += 1
            
            # Update market events
            if self.config.get('enable_market_events', True):
                self.market_events.update_events(self.world_state.current_time, 
                                               {k: v.__dict__ for k, v in self.world_state.towns.items()})
            
            # Update world state
            self.world_state.update_world_state(1)
            
            # Generate trade opportunities (disabled in interactive mode)
            if not hasattr(self, '_interactive_mode') or not self._interactive_mode:
                self._generate_trade_opportunities()
            
            # Execute trades
            self._execute_trades()
            
            # Update statistics
            self._update_statistics()
            
            # Save trade logs
            if self.config.get('save_trade_logs', True):
                self._save_trade_logs()
            
            # Check for simulation end conditions
            if self.current_day >= self.max_days:
                self.stop_simulation()
                break
    
    def _generate_trade_opportunities(self):
        """Generate trade opportunities for caravans"""
        # Get all towns
        towns_data = {}
        for country, town in self.world_state.towns.items():
            towns_data[country] = {
                'country': town.country,
                'region': town.region,
                'population': town.population,
                'gdp_per_capita': town.gdp_per_capita,
                'tech_level': town.tech_level,
                'stability': town.stability,
                'latitude': town.latitude,
                'longitude': town.longitude,
                'inventory': town.inventory.copy(),
                'prices': town.prices.copy(),
                'demand': town.demand.copy(),
                'supply': town.supply.copy()
            }
        
        # Generate opportunities for each item type
        for item_type in self.world_state.items.keys():
            opportunities = self.economy_model.calculate_arbitrage_opportunities(towns_data, item_type)
            
            # Filter opportunities based on decision model
            for opportunity in opportunities[:10]:  # Limit to top 10 opportunities
                origin_town = towns_data[opportunity['origin']]
                dest_town = towns_data[opportunity['destination']]
                
                decision = self.decision_model.evaluate_trade_opportunity(
                    origin_town, dest_town, item_type, self.economy_model
                )
                
                if decision:
                    # Create caravan if it doesn't exist
                    caravan = self.world_state.create_caravan(decision.origin, decision.destination)
                    if caravan:
                        # Load cargo
                        origin_town_obj = self.world_state.towns[decision.origin]
                        origin_price = origin_town_obj.prices.get(item_type, 0)
                        caravan.load_cargo(item_type, decision.quantity, origin_price)
                        
                        # Execute trade
                        trade_cost = origin_town_obj.trade(item_type, decision.quantity, True)
                        if trade_cost > 0:
                            self.total_trades += 1
                            self.total_profit += decision.expected_profit
    
    def _execute_trades(self):
        """Execute trades for active caravans"""
        completed_caravans = []
        
        for caravan_id, caravan in self.world_state.caravans.items():
            if not caravan.is_active:
                continue
            
            # Check if caravan has reached destination
            if caravan.route_progress >= 1.0:
                # Execute trade at destination
                dest_town = self.world_state.towns[caravan.destination.country]
                origin_town = self.world_state.towns[caravan.origin.country]
                
                trade_successful = False
                total_trade_value = 0.0
                
                for item_type, quantity in caravan.cargo.items():
                    if quantity > 0:
                        # Sell items at destination
                        sale_value = dest_town.trade(item_type, quantity, False)
                        
                        if sale_value > 0:
                            trade_successful = True
                            total_trade_value += sale_value
                            
                            # Record trade
                            trade_data = {
                                'timestamp': self.world_state.current_time.isoformat(),
                                'origin': caravan.origin.country,
                                'destination': caravan.destination.country,
                                'item_type': item_type,
                                'quantity': quantity,
                                'price_per_unit': dest_town.prices.get(item_type, 0),
                                'total_value': sale_value,
                                'caravan_id': caravan_id,
                                'risk_level': caravan.risk_level,
                                'travel_time': caravan.travel_time
                            }
                            
                            self.trade_history.append(trade_data)
                            
                            # Update statistics
                            self.total_profit += sale_value - (quantity * caravan.origin.prices.get(item_type, 0))
                
                # Update reputation based on successful trade
                if trade_successful:
                    self._update_reputation_from_trade(origin_town, dest_town, total_trade_value)
                
                completed_caravans.append(caravan_id)
        
        # Remove completed caravans
        for caravan_id in completed_caravans:
            del self.world_state.caravans[caravan_id]
    
    def _update_reputation_from_trade(self, origin_town: Town, dest_town: Town, trade_value: float):
        """Update reputation between towns based on successful trade"""
        # Calculate reputation change based on trade value and existing relationship
        base_reputation_change = 0.01  # Small positive change for successful trade
        
        # Adjust based on trade value (larger trades have more impact)
        value_multiplier = min(trade_value / 10000.0, 2.0)  # Cap at 2x multiplier
        reputation_change = base_reputation_change * value_multiplier
        
        # Adjust based on existing relationship
        current_reputation = origin_town.get_reputation_with(dest_town.country)
        
        if current_reputation >= 0.5:  # Ally
            # Smaller positive change for allies (already good relationship)
            reputation_change *= 0.5
        elif current_reputation <= -0.3:  # Enemy
            # Larger positive change for enemies (improving relationship)
            reputation_change *= 1.5
        elif current_reputation <= -0.1:  # Unfriendly
            # Moderate positive change for unfriendly nations
            reputation_change *= 1.2
        
        # Apply reputation change (bidirectional)
        origin_town.update_reputation(dest_town.country, reputation_change)
        dest_town.update_reputation(origin_town.country, reputation_change)
        
        # Log reputation change if significant
        if abs(reputation_change) > 0.02:
            print(f"Reputation change: {origin_town.country} -> {dest_town.country}: {reputation_change:+.3f}")
    
    def _update_statistics(self):
        """Update daily statistics"""
        stats = {
            'day': self.current_day,
            'timestamp': self.world_state.current_time.isoformat(),
            'total_towns': len(self.world_state.towns),
            'active_caravans': len([c for c in self.world_state.caravans.values() if c.is_active]),
            'total_trades': self.total_trades,
            'total_profit': self.total_profit,
            'global_gdp': self.world_state.global_gdp,
            'global_trade_volume': self.world_state.global_trade_volume,
            'active_market_events': len(self.market_events.active_events)
        }
        
        # Add price statistics
        for item_type in self.world_state.items.keys():
            prices = [town.prices.get(item_type, 0) for town in self.world_state.towns.values()]
            if prices:
                stats[f'{item_type}_avg_price'] = np.mean(prices)
                stats[f'{item_type}_price_volatility'] = np.std(prices)
        
        self.daily_stats.append(stats)
    
    def _save_trade_logs(self):
        """Save trade logs to CSV"""
        if self.trade_history:
            # Convert to DataFrame and save
            df = pd.DataFrame(self.trade_history)
            df.to_csv('data/trade_logs.csv', index=False, mode='a', header=False)
            self.trade_history = []  # Clear after saving
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else timedelta(0)
        
        print(f"\nSimulation completed!")
        print(f"Duration: {duration}")
        print(f"Total days simulated: {self.current_day}")
        print(f"Total trades executed: {self.total_trades}")
        print(f"Total profit: ${self.total_profit:.2f}")
        print(f"Final global GDP: ${self.world_state.global_gdp:,.0f}")
    
    def get_simulation_summary(self) -> Dict:
        """Get a summary of the simulation"""
        return {
            'current_day': self.current_day,
            'total_towns': len(self.world_state.towns),
            'total_items': len(self.world_state.items),
            'active_caravans': len([c for c in self.world_state.caravans.values() if c.is_active]),
            'total_trades': self.total_trades,
            'total_profit': self.total_profit,
            'global_gdp': self.world_state.global_gdp,
            'global_trade_volume': self.world_state.global_trade_volume,
            'active_market_events': len(self.market_events.active_events),
            'simulation_duration': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }
    
    def get_town_statistics(self, country: str) -> Optional[Dict]:
        """Get detailed statistics for a specific town"""
        if country not in self.world_state.towns:
            return None
        
        town = self.world_state.towns[country]
        
        # Calculate trade volume for this town
        town_trades = [t for t in self.trade_history if t['origin'] == country or t['destination'] == country]
        trade_volume = sum(t['total_value'] for t in town_trades)
        
        return {
            'country': town.country,
            'region': town.region,
            'population': town.population,
            'gdp_per_capita': town.gdp_per_capita,
            'tech_level': town.tech_level,
            'stability': town.stability,
            'inventory': town.inventory.copy(),
            'prices': town.prices.copy(),
            'trade_volume': trade_volume,
            'total_trades': len(town_trades)
        }
    
    def get_market_analysis(self) -> Dict:
        """Get market analysis and insights"""
        analysis = {
            'price_trends': {},
            'arbitrage_opportunities': {},
            'market_efficiency': {},
            'risk_analysis': {}
        }
        
        # Analyze price trends for each item
        for item_type in self.world_state.items.keys():
            prices = [town.prices.get(item_type, 0) for town in self.world_state.towns.values()]
            if prices:
                analysis['price_trends'][item_type] = {
                    'mean_price': np.mean(prices),
                    'std_price': np.std(prices),
                    'min_price': np.min(prices),
                    'max_price': np.max(prices),
                    'price_range': np.max(prices) - np.min(prices)
                }
        
        # Get arbitrage opportunities
        towns_data = {k: v.__dict__ for k, v in self.world_state.towns.items()}
        for item_type in self.world_state.items.keys():
            opportunities = self.economy_model.calculate_arbitrage_opportunities(towns_data, item_type)
            analysis['arbitrage_opportunities'][item_type] = opportunities[:5]  # Top 5
        
        # Calculate market efficiency
        for item_type in self.world_state.items.keys():
            efficiency = self.economy_model.get_market_efficiency_score(towns_data, item_type)
            analysis['market_efficiency'][item_type] = efficiency
        
        return analysis
    
    def export_data(self, filename: str = None):
        """Export simulation data to JSON"""
        if not filename:
            filename = f"simulation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'simulation_summary': self.get_simulation_summary(),
            'daily_statistics': self.daily_stats,
            'market_analysis': self.get_market_analysis(),
            'town_statistics': {
                country: self.get_town_statistics(country)
                for country in self.world_state.towns.keys()
            },
            'market_events': [
                {
                    'type': event.event_type,
                    'item_type': event.item_type,
                    'country': event.country,
                    'price_multiplier': event.price_multiplier,
                    'description': event.description,
                    'start_time': event.start_time.isoformat(),
                    'expiry_time': event.expiry_time.isoformat()
                }
                for event in self.market_events.active_events
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Simulation data exported to {filename}")
    
    def run_interactive_mode(self):
        """Run the simulation in interactive mode with console-based interface"""
        import os
        import time
        
        print("QUANT-STYLE ECONOMIC SIMULATOR - INTERACTIVE MODE")
        print("=" * 60)
        print("INTERACTIVE SIMULATION MODE")
        print("=" * 60)
        
        # Set interactive mode flag to disable auto-caravan generation
        self._interactive_mode = True
        
        # Initialize simulation
        self.start_simulation()
        
        # Interactive state
        paused = False
        
        # Clear screen function
        def clear_screen():
            os.system('cls' if os.name == 'nt' else 'clear')
        
        # Dashboard display
        def display_dashboard():
            clear_screen()
            print("QUANT-STYLE ECONOMIC SIMULATOR - INTERACTIVE MODE")
            print("=" * 60)
            
            # Current status
            summary = self.get_simulation_summary()
            print(f"Day: {summary['current_day']:>4} | "
                  f"Status: {'PAUSED' if paused else 'RUNNING'} | "
                  f"Caravans: {summary['active_caravans']:>2} | "
                  f"Profit: ${summary['total_profit']:>12,.2f}")
            
            print("-" * 60)
            
            # Market overview
            analysis = self.get_market_analysis()
            print("MARKET OVERVIEW:")
            for item_type, trends in analysis['price_trends'].items():
                price = trends['mean_price']
                change = trends.get('price_change', 0)
                change_symbol = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
                print(f"  {item_type:15} ${price:>8.2f} {change_symbol} {change:>+6.1f}%")
            
            print("-" * 60)
            
            # Recent trades
            recent_trades = self.trade_history[-5:] if self.trade_history else []
            if recent_trades:
                print("RECENT TRADES:")
                for trade in recent_trades:
                    print(f"  {trade['origin']:>12} → {trade['destination']:<12} "
                          f"{trade['item_type']:>12} ${trade['total_value']:>8,.0f}")
            
            print("-" * 60)
        
        # Show trade opportunities
        def show_trade_opportunities():
            clear_screen()
            print("TRADE OPPORTUNITIES")
            print("=" * 60)
            
            towns_data = {k: v.__dict__ for k, v in self.world_state.towns.items()}
            
            all_opportunities = []
            for item_type in self.world_state.items.keys():
                opportunities = self.economy_model.calculate_arbitrage_opportunities(towns_data, item_type)
                for opp in opportunities:
                    opp['item_type'] = item_type
                    all_opportunities.append(opp)
            
            # Sort by profit margin
            all_opportunities.sort(key=lambda x: x.get('potential_profit_margin', 0), reverse=True)
            
            if not all_opportunities:
                print("No trade opportunities available at the moment.")
                input("\nPress ENTER to continue...")
                return
            
            print("TOP TRADE OPPORTUNITIES:")
            print("-" * 60)
            
            for i, opp in enumerate(all_opportunities[:10], 1):
                profit_margin = opp.get('potential_profit_margin', 0)
                risk_level = opp.get('risk_level', 0.5)  # Default risk level
                print(f"{i:2}. {opp['item_type']:>12} | {opp['origin']:>12} → {opp['destination']:<12} | "
                      f"Profit: {profit_margin:>6.1f}% | Risk: {risk_level:>5.1f}")
            
            print("\nPLAYER ACTIONS:")
            print("  Enter opportunity number to send caravan")
            print("  Press ENTER to continue without action")
            
            choice = input("\nYour choice: ").strip()
            
            if choice.isdigit():
                try:
                    opp_index = int(choice) - 1
                    if 0 <= opp_index < len(all_opportunities):
                        selected_opp = all_opportunities[opp_index]
                        
                        # Get optimal quantity
                        origin_town = self.world_state.towns[selected_opp['origin']]
                        dest_town = self.world_state.towns[selected_opp['destination']]
                        
                        optimal_quantity = self.economy_model.calculate_optimal_trade_quantity(
                            origin_town.__dict__, dest_town.__dict__, selected_opp['item_type']
                        )
                        
                        print(f"\nSending caravan for {selected_opp['item_type']}")
                        print(f"   Route: {selected_opp['origin']} → {selected_opp['destination']}")
                        print(f"   Quantity: {optimal_quantity}")
                        print(f"   Expected Profit: {selected_opp['potential_profit_margin']:.1f}%")
                        
                        # Create caravan
                        caravan = self.world_state.create_caravan(selected_opp['origin'], selected_opp['destination'])
                        if caravan:
                            caravan.load_cargo(selected_opp['item_type'], optimal_quantity, origin_town.prices.get(selected_opp['item_type'], 0))
                            print(f"Caravan {caravan.caravan_id[:8]}... created and dispatched!")
                        else:
                            print("Failed to create caravan")
                    else:
                        print("Invalid opportunity number")
                except ValueError:
                    print("Invalid input")
            
            input("\nPress ENTER to continue...")
        
        # Show caravan status
        def show_caravan_status():
            clear_screen()
            print("CARAVAN STATUS")
            print("=" * 60)
            
            active_caravans = [c for c in self.world_state.caravans.values() if c.is_active]
            
            if not active_caravans:
                print("No active caravans")
            else:
                for i, caravan in enumerate(active_caravans, 1):
                    status = "ACTIVE" if caravan.is_active else "INACTIVE"
                    print(f"{i}. {status} Caravan {caravan.caravan_id[:8]}...")
                    print(f"    Route: {caravan.origin.country:>12} → {caravan.destination.country:<12}")
                    
                    # Show cargo information
                    if caravan.cargo:
                        for item_type, quantity in caravan.cargo.items():
                            print(f"    Cargo: {item_type:>12} | Quantity: {quantity:>6}")
                    else:
                        print(f"    Cargo: {'Empty':>12} | Quantity: {0:>6}")
                    
                    print(f"    Progress: {caravan.route_progress*100:>3.0f}% | Risk: {caravan.risk_level:>5.1f}")
                    print(f"    Travel Time: {caravan.travel_time} days | Distance: {caravan.distance:.0f} km")
                    print()
            
            input("\nPress ENTER to continue...")
        
        # Show price trends
        def show_price_trends():
            clear_screen()
            print("PRICE TRENDS")
            print("=" * 60)
            
            for item_type in self.world_state.items.keys():
                if item_type in self.price_history and self.price_history[item_type]:
                    prices = self.price_history[item_type]
                    if len(prices) >= 2:
                        current_price = prices[-1]
                        prev_price = prices[-2]
                        change = ((current_price - prev_price) / prev_price) * 100
                        change_symbol = "↗️" if change > 0 else "↘️" if change < 0 else "➡️"
                        print(f"{item_type:15} ${current_price:>8.2f} {change_symbol} {change:>+6.1f}%")
                    else:
                        print(f"{item_type:15} ${prices[-1]:>8.2f} ➡️   0.0%")
            
            input("\nPress ENTER to continue...")
        
        # Show reputation matrix
        def show_reputation_matrix():
            clear_screen()
            print("REPUTATION MATRIX")
            print("=" * 80)
            
            countries = list(self.world_state.towns.keys())
            
            # Show all countries in a more compact format
            print(" " * 15, end="")
            for country in countries:
                print(f"{country:>8}", end="")
            print()
            
            for country1 in countries:
                print(f"{country1:15}", end="")
                for country2 in countries:
                    if country1 == country2:
                        print("    -   ", end="")
                    else:
                        rep = self.world_state.towns[country1].get_reputation_with(country2)
                        if rep > 0.5:
                            symbol = "🟢"
                        elif rep < -0.5:
                            symbol = "🔴"
                        else:
                            symbol = "🟡"
                        print(f"{symbol} {rep:>4.1f}", end="")
                print()
            
            print("\n🟢 = Ally (>0.5) | 🟡 = Neutral (-0.1 to 0.1) | 🔴 = Enemy (<-0.5)")
            input("\nPress ENTER to continue...")
        
        # Show arbitrage opportunities
        def show_arbitrage_opportunities():
            clear_screen()
            print("ARBITRAGE OPPORTUNITIES")
            print("=" * 60)
            
            towns_data = {k: v.__dict__ for k, v in self.world_state.towns.items()}
            
            all_opportunities = []
            for item_type in self.world_state.items.keys():
                opportunities = self.economy_model.calculate_arbitrage_opportunities(towns_data, item_type)
                for opp in opportunities:
                    opp['item_type'] = item_type
                    all_opportunities.append(opp)
            
            # Sort by profit margin
            all_opportunities.sort(key=lambda x: x.get('potential_profit_margin', 0), reverse=True)
            
            for i, opp in enumerate(all_opportunities[:10], 1):
                profit_margin = opp.get('potential_profit_margin', 0)
                risk_level = opp.get('risk_level', 0.5)  # Default risk level
                print(f"{i:2}. {opp['item_type']:>12} | {opp['origin']:>12} → {opp['destination']:<12} | "
                      f"Profit: {profit_margin:>6.1f}% | Risk: {risk_level:>5.1f}")
            
            input("\nPress ENTER to continue...")
        
        # Show detailed town info
        def show_detailed_town_info():
            clear_screen()
            print("DETAILED TOWN INFORMATION")
            print("=" * 60)
            
            # Show list of available towns
            print("AVAILABLE TOWNS:")
            print("-" * 60)
            countries = list(self.world_state.towns.keys())
            for i, country in enumerate(countries, 1):
                town = self.world_state.towns[country]
                print(f"{i:2}. {country:15} | {town.region:12} | Pop: {town.population:>8,} | GDP: ${town.gdp_per_capita:>6,}")
            
            print("\nOPTIONS:")
            print("  Enter town number (1-{}) for detailed info".format(len(countries)))
            print("  Enter town name for detailed info")
            print("  Press ENTER to continue")
            
            choice = input("\nYour choice: ").strip()
            
            if choice:
                selected_town = None
                
                # Check if it's a number
                if choice.isdigit():
                    try:
                        town_index = int(choice) - 1
                        if 0 <= town_index < len(countries):
                            selected_town = self.world_state.towns[countries[town_index]]
                    except (ValueError, IndexError):
                        pass
                else:
                    # Check if it's a town name
                    if choice in self.world_state.towns:
                        selected_town = self.world_state.towns[choice]
                
                if selected_town:
                    clear_screen()
                    print(f"{selected_town.country.upper()} - DETAILED INFORMATION")
                    print("=" * 60)
                    
                    print(f"BASIC INFO:")
                    print(f"  Region: {selected_town.region}")
                    print(f"  Population: {selected_town.population:,}")
                    print(f"  GDP per capita: ${selected_town.gdp_per_capita:,}")
                    print(f"  Tech Level: {selected_town.tech_level}")
                    print(f"  Stability: {selected_town.stability:.2f}")
                    
                    print(f"\nPRODUCTION FOCUS:")
                    for item_type, focus in selected_town.production_focus.items():
                        if focus > 1.5:
                            focus_symbol = "HIGH"
                        elif focus > 1.2:
                            focus_symbol = "MED"
                        else:
                            focus_symbol = "LOW"
                        print(f"  {focus_symbol:>3} {item_type:>15}: {focus:>6.2f}x")
                    
                    print(f"\nREPUTATION:")
                    allies = [c for c, r in selected_town.reputation.items() if r > 0.5]
                    enemies = [c for c, r in selected_town.reputation.items() if r < -0.5]
                    neutral = [c for c, r in selected_town.reputation.items() if -0.1 <= r <= 0.1]
                    
                    if allies:
                        print(f"  Allies: {', '.join(allies)}")
                    if enemies:
                        print(f"  Enemies: {', '.join(enemies)}")
                    if neutral:
                        print(f"  Neutral: {', '.join(neutral[:5])}{'...' if len(neutral) > 5 else ''}")
                    
                    print(f"\nCURRENT PRICES:")
                    for item_type, price in selected_town.prices.items():
                        # Compare with average price
                        avg_price = np.mean([t.prices.get(item_type, 0) for t in self.world_state.towns.values()])
                        price_diff = ((price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
                        price_symbol = "↗️" if price_diff > 5 else "↘️" if price_diff < -5 else "➡️"
                        print(f"  {price_symbol} {item_type:>15}: ${price:>8.2f} ({price_diff:>+6.1f}%)")
                    
                    print(f"\nINVENTORY:")
                    for item_type, quantity in selected_town.inventory.items():
                        if quantity > 0:
                            print(f"  {item_type:>15}: {quantity:>8,}")
                    
                    print(f"\nSUPPLY/DEMAND:")
                    for item_type in self.world_state.items.keys():
                        supply = selected_town.supply.get(item_type, 0)
                        demand = selected_town.demand.get(item_type, 0)
                        if supply > 0 or demand > 0:
                            ratio = supply / demand if demand > 0 else float('inf')
                            if ratio > 1.2:
                                ratio_symbol = "🟢"
                            elif ratio > 0.8:
                                ratio_symbol = "🟡"
                            else:
                                ratio_symbol = "🔴"
                            print(f"  {ratio_symbol} {item_type:>15}: Supply {supply:>6,} | Demand {demand:>6,} | Ratio {ratio:>5.2f}")
                
                else:
                    print(f"Town '{choice}' not found")
            
            input("\nPress ENTER to continue...")
        
        # Show manual caravan creation
        def show_manual_caravan_creation():
            clear_screen()
            print("MANUAL CARAVAN CREATION")
            print("=" * 60)
            
            # Show available towns
            print("AVAILABLE TOWNS:")
            print("-" * 60)
            countries = list(self.world_state.towns.keys())
            for i, country in enumerate(countries, 1):
                town = self.world_state.towns[country]
                print(f"{i:2}. {country:15} | {town.region:12} | Pop: {town.population:>8,}")
            
            print("\nCREATE CARAVAN:")
            print("  Format: origin_number,destination_number,item_type,quantity")
            print("  Example: 1,5,Tech,100")
            print("  Press ENTER to cancel")
            
            choice = input("\nYour choice: ").strip()
            
            if choice:
                try:
                    parts = choice.split(',')
                    if len(parts) == 4:
                        origin_idx = int(parts[0]) - 1
                        dest_idx = int(parts[1]) - 1
                        item_type = parts[2].strip()
                        quantity = int(parts[3])
                        
                        if (0 <= origin_idx < len(countries) and 
                            0 <= dest_idx < len(countries) and
                            item_type in self.world_state.items.keys() and
                            quantity > 0):
                            
                            origin_country = countries[origin_idx]
                            dest_country = countries[dest_idx]
                            
                            if origin_country == dest_country:
                                print("Origin and destination cannot be the same")
                            else:
                                # Check if we have enough inventory
                                origin_town = self.world_state.towns[origin_country]
                                available = origin_town.inventory.get(item_type, 0)
                                
                                if available >= quantity:
                                    # Create caravan
                                    caravan = self.world_state.create_caravan(origin_country, dest_country)
                                    if caravan:
                                        # Load cargo
                                        origin_price = origin_town.prices.get(item_type, 0)
                                        caravan.load_cargo(item_type, quantity, origin_price)
                                        
                                        # Execute the trade (buy from origin)
                                        trade_cost = origin_town.trade(item_type, quantity, True)
                                        
                                        if trade_cost > 0:
                                            print(f"Caravan {caravan.caravan_id[:8]}... created!")
                                            print(f"   Route: {origin_country} → {dest_country}")
                                            print(f"   Cargo: {quantity} {item_type}")
                                            print(f"   Value: ${quantity * origin_price:,.2f}")
                                            print(f"   Travel Time: {caravan.travel_time} days")
                                            print(f"   Risk Level: {caravan.risk_level:.2f}")
                                        else:
                                            print("Failed to execute trade at origin")
                                            # Remove the caravan if trade failed
                                            del self.world_state.caravans[caravan.caravan_id]
                                    else:
                                        print("Failed to create caravan")
                                else:
                                    print(f"Insufficient inventory. Available: {available}, Requested: {quantity}")
                        else:
                            print("Invalid input format or values")
                    else:
                        print("Invalid format. Use: origin,destination,item,quantity")
                except (ValueError, IndexError) as e:
                    print(f"Invalid input: {e}")
            
            input("\nPress ENTER to continue...")
        
        # Main interactive loop
        try:
            while True:
                display_dashboard()
                
                print("\nCONTROLS:")
                print("  1 - Step forward 1 day")
                print("  7 - Step forward 7 days")
                print("  30 - Step forward 30 days")
                print("  T - Show trade opportunities (send caravans)")
                print("  M - Manual caravan creation")
                print("  C - Show caravan status")
                print("  P - Show price trends")
                print("  R - Show reputation matrix")
                print("  A - Show arbitrage opportunities")
                print("  D - Show detailed town info")
                print("  H - Show help")
                print("  Q - Quit simulation")
                
                choice = input("\nEnter your choice: ").strip().upper()
                
                if choice == '1':
                    self.run_simulation_step(1)
                    print(f"Stepped forward 1 day (Day {self.current_day})")
                
                elif choice == '7':
                    self.run_simulation_step(7)
                    print(f"Stepped forward 7 days (Day {self.current_day})")
                
                elif choice == '30':
                    self.run_simulation_step(30)
                    print(f"Stepped forward 30 days (Day {self.current_day})")
                
                elif choice == 'T':
                    show_trade_opportunities()
                
                elif choice == 'M':
                    show_manual_caravan_creation()
                
                elif choice == 'C':
                    show_caravan_status()
                
                elif choice == 'P':
                    show_price_trends()
                
                elif choice == 'R':
                    show_reputation_matrix()
                
                elif choice == 'A':
                    show_arbitrage_opportunities()
                
                elif choice == 'D':
                    show_detailed_town_info()
                
                elif choice == 'H':
                    clear_screen()
                    print("QUANT-STYLE ECONOMIC SIMULATOR - INTERACTIVE MODE")
                    print("=" * 60)
                    print("CONTROLS:")
                    print("  1        - Step forward 1 day")
                    print("  7        - Step forward 7 days")
                    print("  30       - Step forward 30 days")
                    print("  T        - Show trade opportunities (send caravans)")
                    print("  M        - Manual caravan creation")
                    print("  C        - Show caravan status")
                    print("  P        - Show price trends")
                    print("  R        - Show reputation matrix")
                    print("  A        - Show arbitrage opportunities")
                    print("  D        - Show detailed town info")
                    print("  H        - Show this help")
                    print("  Q        - Quit simulation")
                    print("=" * 60)
                    print("\nTRADE OPPORTUNITIES (T):")
                    print("  - View top 10 profitable trade opportunities")
                    print("  - Select by number to automatically send caravan")
                    print("  - System calculates optimal quantity")
                    print("\nMANUAL CARAVAN (M):")
                    print("  - Create custom caravans with specific routes")
                    print("  - Format: origin,destination,item,quantity")
                    print("  - Example: 1,5,Tech,100")
                    print("\nTOWN INFO (D):")
                    print("  - View list of all available towns")
                    print("  - Select by number or name for detailed info")
                    print("  - See prices, inventory, reputation, and more")
                    input("\nPress ENTER to continue...")
                
                elif choice == 'Q':
                    print("\nQuitting simulation...")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted")
        
        finally:
            if self.is_running:
                self.stop_simulation()
            print("\nFinal Statistics:")
            summary = self.get_simulation_summary()
            print(f"  Total Days: {summary['current_day']}")
            print(f"  Total Trades: {summary['total_trades']}")
            print(f"  Total Profit: ${summary['total_profit']:,.2f}")
            print(f"  Final GDP: ${summary['global_gdp']:,.0f}") 
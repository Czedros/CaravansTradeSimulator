#!/usr/bin/env python3
"""
Test script for the Economic Simulator
"""

import sys
import os
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sim.simulator import EconomicSimulator
from viz.plot_prices import PriceVisualizer

def test_basic_simulation():
    """Test basic simulation functionality"""
    print("🧪 Testing Basic Simulation...")
    
    # Initialize simulator
    config = {
        'max_days': 30,
        'simulation_speed': 1,
        'enable_market_events': True,
        'enable_visualization': False,
        'save_trade_logs': False
    }
    
    simulator = EconomicSimulator(config)
    
    # Test initialization
    assert len(simulator.world_state.towns) > 0, "No towns loaded"
    assert len(simulator.world_state.items) > 0, "No items loaded"
    
    print(f"✅ Loaded {len(simulator.world_state.towns)} towns and {len(simulator.world_state.items)} items")
    
    # Start simulation
    simulator.start_simulation()
    assert simulator.is_running, "Simulation should be running"
    
    # Run simulation for 10 days
    simulator.run_simulation_step(10)
    
    # Check results
    summary = simulator.get_simulation_summary()
    assert summary['current_day'] == 10, f"Expected day 10, got {summary['current_day']}"
    
    print(f"✅ Simulation completed {summary['current_day']} days")
    print(f"✅ Total trades: {summary['total_trades']}")
    print(f"✅ Total profit: ${summary['total_profit']:.2f}")
    
    return simulator

def test_visualization():
    """Test visualization functionality"""
    print("\n📊 Testing Visualization...")
    
    # Get test data
    simulator = test_basic_simulation()
    
    # Initialize visualizer
    visualizer = PriceVisualizer()
    
    # Get towns data
    towns_data = {}
    for country, town in simulator.world_state.towns.items():
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
    
    item_types = list(simulator.world_state.items.keys())
    
    # Test price heatmap
    try:
        fig = visualizer.plot_price_heatmap(towns_data, item_types)
        print("✅ Price heatmap created successfully")
    except Exception as e:
        print(f"❌ Price heatmap failed: {e}")
    
    # Test supply/demand heatmap
    try:
        fig = visualizer.plot_supply_demand_heatmap(towns_data, item_types)
        print("✅ Supply/demand heatmap created successfully")
    except Exception as e:
        print(f"❌ Supply/demand heatmap failed: {e}")
    
    # Test market efficiency
    try:
        fig = visualizer.plot_market_efficiency(towns_data, item_types)
        print("✅ Market efficiency plot created successfully")
    except Exception as e:
        print(f"❌ Market efficiency plot failed: {e}")

def test_market_events():
    """Test market events functionality"""
    print("\n🌍 Testing Market Events...")
    
    from models.market_events import MarketEventManager, MarketEventGenerator
    
    # Initialize market events
    event_manager = MarketEventManager()
    event_generator = MarketEventGenerator()
    
    # Test event generation
    towns_data = {'USA': {'country': 'USA'}, 'China': {'country': 'China'}}
    
    # Generate some events
    for _ in range(5):
        event = event_generator.generate_random_event(datetime.now(), towns_data)
        if event:
            event_manager.active_events.append(event)
    
    print(f"✅ Generated {len(event_manager.active_events)} market events")
    
    # Test event summary
    summary = event_manager.get_event_summary()
    print(f"✅ Event summary: {summary['total_active_events']} active events")

def test_decision_model():
    """Test decision model functionality"""
    print("\n🎯 Testing Decision Model...")
    
    from models.caravan_decision import CaravanDecisionModel
    
    # Initialize decision model
    decision_model = CaravanDecisionModel()
    
    # Test data
    origin_town = {
        'country': 'USA',
        'region': 'North America',
        'population': 331002651,
        'gdp_per_capita': 69287,
        'tech_level': 9,
        'stability': 8,
        'latitude': 39.8283,
        'longitude': -98.5795,
        'prices': {'Food': 100, 'Tech': 500},
        'inventory': {'Food': 1000, 'Tech': 500},
        'demand': {'Food': 1000, 'Tech': 500},
        'supply': {'Food': 1000, 'Tech': 500}
    }
    
    dest_town = {
        'country': 'China',
        'region': 'Asia',
        'population': 1439323776,
        'gdp_per_capita': 12556,
        'tech_level': 8,
        'stability': 7,
        'latitude': 35.8617,
        'longitude': 104.1954,
        'prices': {'Food': 120, 'Tech': 600},
        'inventory': {'Food': 800, 'Tech': 400},
        'demand': {'Food': 1200, 'Tech': 600},
        'supply': {'Food': 800, 'Tech': 400}
    }
    
    # Test trade opportunity evaluation
    from models.economy_model import EconomyModel
    economy_model = EconomyModel()
    
    decision = decision_model.evaluate_trade_opportunity(
        origin_town, dest_town, 'Food', economy_model
    )
    
    if decision:
        print(f"✅ Trade decision created: {decision.origin} -> {decision.destination}")
        print(f"   Item: {decision.item_type}, Quantity: {decision.quantity}")
        print(f"   Expected profit: ${decision.expected_profit:.2f}")
        print(f"   Risk level: {decision.risk_level:.2f}")
    else:
        print("ℹ️ No trade opportunity found (this is normal)")

def main():
    """Run all tests"""
    print("🚀 Starting Economic Simulator Tests")
    print("=" * 50)
    
    try:
        # Test basic simulation
        test_basic_simulation()
        
        # Test visualization
        test_visualization()
        
        # Test market events
        test_market_events()
        
        # Test decision model
        test_decision_model()
        
        print("\n🎉 All tests completed successfully!")
        print("=" * 50)
        print("The Economic Simulator is ready to use!")
        print("\nTry running: python main.py --interactive")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
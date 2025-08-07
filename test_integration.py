#!/usr/bin/env python3
"""Integration test for reputation and production focus systems"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sim.entities import Town
from models.caravan_decision import CaravanDecisionModel
from models.economy_model import EconomyModel

def test_integration():
    """Test the full integration of reputation and production focus systems"""
    print("Testing Full Integration...")
    
    # Create towns with reputation and production focus
    usa = Town('USA', 'North America', 331002651, 69287, 9, 8, 39.8283, -98.5795)
    china = Town('China', 'Asia', 1439323776, 12556, 8, 7, 35.8617, 104.1954)
    uk = Town('UK', 'Europe', 67886011, 42328, 9, 8, 55.3781, -3.4360)
    
    # Test reputation system
    print(f"\n1. Reputation System Test:")
    print(f"USA-UK reputation: {usa.get_reputation_with('UK'):.3f} ({usa.get_reputation_level('UK')})")
    print(f"USA-China reputation: {usa.get_reputation_with('China'):.3f} ({usa.get_reputation_level('China')})")
    print(f"USA is ally with UK: {usa.is_ally('UK')}")
    print(f"USA is enemy with China: {usa.is_enemy('China')}")
    
    # Test production focus system
    print(f"\n2. Production Focus System Test:")
    print(f"USA Tech focus: {usa.get_production_focus_multiplier('Tech'):.1f}x")
    print(f"China Tech focus: {china.get_production_focus_multiplier('Tech'):.1f}x")
    print(f"USA Armaments focus: {usa.get_production_focus_multiplier('Armaments'):.1f}x")
    print(f"China Raw Resource focus: {china.get_production_focus_multiplier('Raw Resource'):.1f}x")
    
    # Test decision model integration
    print(f"\n3. Decision Model Integration Test:")
    decision_model = CaravanDecisionModel()
    
    # Create town data dictionaries for decision model
    usa_data = {
        'country': 'USA',
        'prices': {'Tech': 500, 'Armaments': 300},
        'inventory': {'Tech': 1000, 'Armaments': 800},
        'demand': {'Tech': 1200, 'Armaments': 900},
        'stability': 8,
        'latitude': 39.8283,
        'longitude': -98.5795,
        'production_focus': usa.production_focus,
        'reputation': usa.reputation
    }
    
    uk_data = {
        'country': 'UK',
        'prices': {'Tech': 600, 'Armaments': 350},
        'inventory': {'Tech': 800, 'Armaments': 600},
        'demand': {'Tech': 1000, 'Armaments': 700},
        'stability': 8,
        'latitude': 55.3781,
        'longitude': -3.4360,
        'production_focus': uk.production_focus,
        'reputation': uk.reputation
    }
    
    # Test trade opportunity evaluation
    opportunity = decision_model.evaluate_trade_opportunity(
        usa_data, uk_data, 'Tech', None
    )
    
    if opportunity:
        print(f"Trade opportunity found: {opportunity.origin} -> {opportunity.destination}")
        print(f"Item: {opportunity.item_type}, Quantity: {opportunity.quantity}")
        print(f"Expected profit: ${opportunity.expected_profit:.2f}")
        print(f"Risk level: {opportunity.risk_level:.3f}")
        print(f"Confidence score: {opportunity.confidence_score:.3f}")
    else:
        print("No trade opportunity found")
    
    # Test economy model integration
    print(f"\n4. Economy Model Integration Test:")
    economy_model = EconomyModel()
    
    # Test arbitrage opportunities
    towns_data = {
        'USA': usa_data,
        'UK': uk_data,
        'China': {
            'country': 'China',
            'prices': {'Tech': 400, 'Armaments': 250},
            'inventory': {'Tech': 1500, 'Armaments': 1000},
            'demand': {'Tech': 2000, 'Armaments': 1200},
            'stability': 7,
            'latitude': 35.8617,
            'longitude': 104.1954,
            'production_focus': china.production_focus,
            'reputation': china.reputation
        }
    }
    
    arbitrage_opportunities = economy_model.calculate_arbitrage_opportunities(towns_data, 'Tech')
    
    if arbitrage_opportunities:
        print(f"Found {len(arbitrage_opportunities)} arbitrage opportunities:")
        for i, opp in enumerate(arbitrage_opportunities[:3]):  # Show first 3
            print(f"  {i+1}. {opp['origin']} -> {opp['destination']}: {opp['potential_profit_margin']:.1f}% profit")
            print(f"     Reputation impact: {opp['reputation_impact']:.3f}")
            print(f"     Production focus impact: {opp['production_focus_impact']:.3f}")
    else:
        print("No arbitrage opportunities found")
    
    # Test reputation updates
    print(f"\n5. Reputation Update Test:")
    initial_reputation = usa.get_reputation_with('China')
    print(f"Initial USA-China reputation: {initial_reputation:.3f}")
    
    # Simulate successful trade
    usa.update_reputation('China', 0.05)
    china.update_reputation('USA', 0.05)
    
    updated_reputation = usa.get_reputation_with('China')
    print(f"After trade (+0.05): {updated_reputation:.3f}")
    
    # Check if status changed
    print(f"USA is enemy with China: {usa.is_enemy('China')}")
    
    print(f"\n✅ All integration tests completed successfully!")

if __name__ == "__main__":
    test_integration()

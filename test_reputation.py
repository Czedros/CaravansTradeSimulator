#!/usr/bin/env python3
"""Test script for reputation and production focus features"""

from sim.entities import Town

def test_reputation_system():
    """Test the reputation system"""
    print("Testing Reputation System...")
    
    # Create a USA town
    usa = Town('USA', 'North America', 331002651, 69287, 9, 8, 39.8283, -98.5795)
    
    # Test reputation with different countries
    print(f"USA reputation with China: {usa.get_reputation_with('China'):.3f}")
    print(f"USA reputation with UK: {usa.get_reputation_with('UK'):.3f}")
    print(f"USA reputation with Russia: {usa.get_reputation_with('Russia'):.3f}")
    
    # Test ally/enemy status
    print(f"USA is ally with UK: {usa.is_ally('UK')}")
    print(f"USA is enemy with Russia: {usa.is_enemy('Russia')}")
    print(f"USA is enemy with China: {usa.is_enemy('China')}")
    
    # Test reputation levels
    print(f"USA reputation level with UK: {usa.get_reputation_level('UK')}")
    print(f"USA reputation level with Russia: {usa.get_reputation_level('Russia')}")
    print(f"USA reputation level with China: {usa.get_reputation_level('China')}")

def test_production_focus():
    """Test the production focus system"""
    print("\nTesting Production Focus System...")
    
    # Create different towns
    usa = Town('USA', 'North America', 331002651, 69287, 9, 8, 39.8283, -98.5795)
    china = Town('China', 'Asia', 1439323776, 12556, 8, 7, 35.8617, 104.1954)
    russia = Town('Russia', 'Europe', 145912025, 11289, 7, 4, 61.5240, 105.3188)
    
    # Test production focus for different items
    items = ['Tech', 'Armaments', 'Raw Resource', 'Energies', 'Food']
    
    for item in items:
        usa_focus = usa.get_production_focus_multiplier(item)
        china_focus = china.get_production_focus_multiplier(item)
        russia_focus = russia.get_production_focus_multiplier(item)
        
        print(f"{item}: USA={usa_focus:.1f}, China={china_focus:.1f}, Russia={russia_focus:.1f}")

def test_reputation_updates():
    """Test reputation updates"""
    print("\nTesting Reputation Updates...")
    
    usa = Town('USA', 'North America', 331002651, 69287, 9, 8, 39.8283, -98.5795)
    china = Town('China', 'Asia', 1439323776, 12556, 8, 7, 35.8617, 104.1954)
    
    # Initial reputation
    initial_reputation = usa.get_reputation_with('China')
    print(f"Initial USA-China reputation: {initial_reputation:.3f}")
    
    # Update reputation
    usa.update_reputation('China', 0.1)
    updated_reputation = usa.get_reputation_with('China')
    print(f"After +0.1 update: {updated_reputation:.3f}")
    
    # Check if status changed
    print(f"USA is enemy with China: {usa.is_enemy('China')}")

if __name__ == "__main__":
    test_reputation_system()
    test_production_focus()
    test_reputation_updates()
    print("\nAll tests completed!")

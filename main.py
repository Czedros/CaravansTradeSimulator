#!/usr/bin/env python3
"""
Quant-style Economic Simulator for Trading (Caravan Game)

A comprehensive economic simulation that models global trade routes,
price dynamics, and market events using quantitative methods.

Usage:
    python main.py [options]

Options:
    --interactive    Run in interactive mode
    --days N         Run simulation for N days (default: 365)
    --speed N        Simulation speed in days per second (default: 1)
    --visualize      Enable visualization
    --export         Export data at the end
    --config FILE    Load configuration from file
"""

import argparse
import sys
import os
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sim.simulator import EconomicSimulator
from viz.plot_prices import PriceVisualizer

def load_config(config_file: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found, using defaults")
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in config file {config_file}, using defaults")
        return {}

def run_simulation(config: dict):
    """Run the economic simulation"""
    print("=" * 60)
    print("Caravans ECONOMIC SIMULATOR")
    print("=" * 60)
    
    # Initialize simulator
    simulator = EconomicSimulator(config)
    
    # Start simulation
    simulator.start_simulation()
    
    # Run simulation
    max_days = config.get('max_days', 365)
    simulation_speed = config.get('simulation_speed', 1)
    
    print(f"\nRunning simulation for {max_days} days...")
    print(f"Simulation speed: {simulation_speed} days/second")
    
    # Run simulation steps
    days_per_step = min(7, max_days)  # Run in weekly chunks
    for day in range(0, max_days, days_per_step):
        remaining_days = min(days_per_step, max_days - day)
        simulator.run_simulation_step(remaining_days)
        
        # Print progress
        progress = (day + remaining_days) / max_days * 100
        print(f"Progress: {progress:.1f}% ({day + remaining_days}/{max_days} days)")
        
        # Show current status
        if (day + remaining_days) % 30 == 0:  # Every 30 days
            summary = simulator.get_simulation_summary()
            print(f"  Active caravans: {summary['active_caravans']}")
            print(f"  Total trades: {summary['total_trades']}")
            print(f"  Total profit: ${summary['total_profit']:.2f}")
    
    # Stop simulation
    simulator.stop_simulation()
    
    return simulator

def create_visualizations(simulator: EconomicSimulator, config: dict):
    """Create and save visualizations"""
    if not config.get('visualize', True):
        return
    
    print("\nCreating visualizations...")
    
    # Initialize visualizer
    visualizer = PriceVisualizer()
    
    # Get data for visualization
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
    
    # Create dashboard
    dashboard_plots = visualizer.create_dashboard(
        towns_data, simulator.daily_stats, simulator.price_history, item_types
    )
    
    # Save plots
    plots_dir = config.get('plots_directory', 'plots')
    visualizer.save_plots(dashboard_plots, plots_dir)
    
    print(f"Visualizations saved to {plots_dir}/")

def export_data(simulator: EconomicSimulator, config: dict):
    """Export simulation data"""
    if not config.get('export', False):
        return
    
    print("\nExporting simulation data...")
    
    # Export to JSON
    export_filename = config.get('export_filename', None)
    simulator.export_data(export_filename)
    
    # Export trade logs to CSV
    if simulator.trade_history:
        import pandas as pd
        df = pd.DataFrame(simulator.trade_history)
        csv_filename = config.get('csv_filename', 'trade_logs.csv')
        df.to_csv(csv_filename, index=False)
        print(f"Trade logs exported to {csv_filename}")

def print_summary(simulator: EconomicSimulator):
    """Print simulation summary"""
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    
    summary = simulator.get_simulation_summary()
    
    print(f"Total Days Simulated: {summary['current_day']}")
    print(f"Total Towns: {summary['total_towns']}")
    print(f"Total Items: {summary['total_items']}")
    print(f"Active Caravans: {summary['active_caravans']}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Total Profit: ${summary['total_profit']:,.2f}")
    print(f"Global GDP: ${summary['global_gdp']:,.0f}")
    print(f"Global Trade Volume: ${summary['global_trade_volume']:,.0f}")
    print(f"Active Market Events: {summary['active_market_events']}")
    
    # Market analysis
    analysis = simulator.get_market_analysis()
    print(f"\nMARKET ANALYSIS:")
    for item_type, trends in analysis['price_trends'].items():
        print(f"  {item_type}:")
        print(f"    Average Price: ${trends['mean_price']:.2f}")
        print(f"    Price Range: ${trends['price_range']:.2f}")
        print(f"    Volatility: {trends['std_price']:.2f}")

def interactive_mode(config: dict):
    """Run interactive mode"""
    print("=" * 60)
    print("Caravans ECONOMIC SIMULATOR - INTERACTIVE MODE")
    print("=" * 60)
    
    simulator = EconomicSimulator(config)
    simulator.run_interactive_mode()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Caravans Economic Simulator for Global Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--days', type=int, default=365,
                       help='Number of days to simulate (default: 365)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Simulation speed in days per second (default: 1.0)')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    parser.add_argument('--export', action='store_true',
                       help='Export data at the end')
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')
    parser.add_argument('--plots-dir', type=str, default='plots',
                       help='Directory to save plots (default: plots)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override with command line arguments
    config.update({
        'max_days': args.days,
        'simulation_speed': args.speed,
        'visualize': args.visualize,
        'export': args.export,
        'plots_directory': args.plots_dir
    })
    
    try:
        if args.interactive:
            interactive_mode(config)
        else:
            # Run simulation
            simulator = run_simulation(config)
            
            # Create visualizations
            create_visualizations(simulator, config)
            
            # Export data
            export_data(simulator, config)
            
            # Print summary
            print_summary(simulator)
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 
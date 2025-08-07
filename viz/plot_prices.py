import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime, timedelta

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PriceVisualizer:
    """Visualization class for price trends and economic data"""
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.dpi = 100
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_price_trends(self, price_data: Dict[str, List[float]], 
                         item_types: List[str], days: List[int] = None,
                         title: str = "Price Trends Over Time"):
        """Plot price trends for multiple item types"""
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        if days is None:
            days = list(range(len(next(iter(price_data.values())))))
        
        for i, item_type in enumerate(item_types):
            if item_type in price_data and price_data[item_type]:
                ax.plot(days, price_data[item_type], 
                       label=item_type, color=self.colors[i % len(self.colors)],
                       linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Price ($)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_price_heatmap(self, towns: Dict[str, Dict], item_types: List[str],
                          title: str = "Price Heatmap by Country and Item"):
        """Create a heatmap of prices across countries and items"""
        # Prepare data for heatmap
        countries = list(towns.keys())
        price_matrix = []
        
        for country in countries:
            row = []
            for item_type in item_types:
                price = towns[country].get('prices', {}).get(item_type, 0)
                row.append(price)
            price_matrix.append(row)
        
        price_matrix = np.array(price_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8), dpi=self.dpi)
        
        im = ax.imshow(price_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(item_types)))
        ax.set_yticks(range(len(countries)))
        ax.set_xticklabels(item_types, rotation=45, ha='right')
        ax.set_yticklabels(countries)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Price ($)', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(countries)):
            for j in range(len(item_types)):
                text = ax.text(j, i, f'${price_matrix[i, j]:.0f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    def plot_supply_demand_heatmap(self, towns: Dict[str, Dict], item_types: List[str],
                                  title: str = "Supply/Demand Heatmap"):
        """Create a heatmap showing supply and demand ratios"""
        countries = list(towns.keys())
        supply_demand_matrix = []
        
        for country in countries:
            row = []
            for item_type in item_types:
                supply = towns[country].get('supply', {}).get(item_type, 1000)
                demand = towns[country].get('demand', {}).get(item_type, 1000)
                ratio = supply / demand if demand > 0 else 1.0
                row.append(ratio)
            supply_demand_matrix.append(row)
        
        supply_demand_matrix = np.array(supply_demand_matrix)
        
        fig, ax = plt.subplots(figsize=(14, 8), dpi=self.dpi)
        
        im = ax.imshow(supply_demand_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
        
        # Set labels
        ax.set_xticks(range(len(item_types)))
        ax.set_yticks(range(len(countries)))
        ax.set_xticklabels(item_types, rotation=45, ha='right')
        ax.set_yticklabels(countries)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Supply/Demand Ratio', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(countries)):
            for j in range(len(item_types)):
                text = ax.text(j, i, f'{supply_demand_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    def plot_trade_network(self, towns: Dict[str, Dict], trade_routes: List[Dict],
                          title: str = "Trade Network"):
        """Plot the trade network between countries"""
        fig, ax = plt.subplots(figsize=(15, 10), dpi=self.dpi)
        
        # Extract coordinates
        countries = list(towns.keys())
        lats = [towns[country].get('latitude', 0) for country in countries]
        lons = [towns[country].get('longitude', 0) for country in countries]
        
        # Plot countries
        ax.scatter(lons, lats, s=100, c='red', alpha=0.7, zorder=5)
        
        # Add country labels
        for i, country in enumerate(countries):
            ax.annotate(country, (lons[i], lats[i]), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, fontweight='bold')
        
        # Plot trade routes
        for route in trade_routes:
            origin = route.get('origin', '')
            destination = route.get('destination', '')
            
            if origin in towns and destination in towns:
                origin_lat = towns[origin].get('latitude', 0)
                origin_lon = towns[origin].get('longitude', 0)
                dest_lat = towns[destination].get('latitude', 0)
                dest_lon = towns[destination].get('longitude', 0)
                
                # Plot route
                ax.plot([origin_lon, dest_lon], [origin_lat, dest_lat], 
                       'b-', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Set world map bounds
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        
        plt.tight_layout()
        return fig
    
    def plot_economic_indicators(self, daily_stats: List[Dict],
                               title: str = "Economic Indicators Over Time"):
        """Plot key economic indicators over time"""
        if not daily_stats:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        days = [stat['day'] for stat in daily_stats]
        
        # Plot 1: Total trades
        trades = [stat.get('total_trades', 0) for stat in daily_stats]
        ax1.plot(days, trades, 'b-', linewidth=2)
        ax1.set_title('Total Trades')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Number of Trades')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total profit
        profits = [stat.get('total_profit', 0) for stat in daily_stats]
        ax2.plot(days, profits, 'g-', linewidth=2)
        ax2.set_title('Total Profit')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Profit ($)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Active caravans
        caravans = [stat.get('active_caravans', 0) for stat in daily_stats]
        ax3.plot(days, caravans, 'r-', linewidth=2)
        ax3.set_title('Active Caravans')
        ax3.set_xlabel('Days')
        ax3.set_ylabel('Number of Caravans')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Global GDP
        gdp = [stat.get('global_gdp', 0) for stat in daily_stats]
        ax4.plot(days, gdp, 'purple', linewidth=2)
        ax4.set_title('Global GDP')
        ax4.set_xlabel('Days')
        ax4.set_ylabel('GDP ($)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_price_volatility(self, price_data: Dict[str, List[float]], 
                            item_types: List[str], days: List[int] = None,
                            title: str = "Price Volatility Analysis"):
        """Plot price volatility for different item types"""
        if not price_data:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        volatilities = []
        item_names = []
        
        for item_type in item_types:
            if item_type in price_data and len(price_data[item_type]) > 1:
                prices = price_data[item_type]
                volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                volatilities.append(volatility)
                item_names.append(item_type)
        
        # Plot 1: Volatility bar chart
        bars = ax1.bar(item_names, volatilities, color=self.colors[:len(item_names)])
        ax1.set_title('Price Volatility by Item Type')
        ax1.set_xlabel('Item Type')
        ax1.set_ylabel('Coefficient of Variation')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, vol in zip(bars, volatilities):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{vol:.3f}', ha='center', va='bottom')
        
        # Plot 2: Price ranges
        price_ranges = []
        for item_type in item_types:
            if item_type in price_data and price_data[item_type]:
                prices = price_data[item_type]
                price_range = max(prices) - min(prices)
                price_ranges.append(price_range)
            else:
                price_ranges.append(0)
        
        bars2 = ax2.bar(item_names, price_ranges, color=self.colors[len(item_names):len(item_names)*2])
        ax2.set_title('Price Range by Item Type')
        ax2.set_xlabel('Item Type')
        ax2.set_ylabel('Price Range ($)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, prange in zip(bars2, price_ranges):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${prange:.0f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def plot_market_efficiency(self, towns: Dict[str, Dict], item_types: List[str],
                             title: str = "Market Efficiency Analysis"):
        """Plot market efficiency scores for different items"""
        efficiency_scores = []
        
        for item_type in item_types:
            prices = []
            for country in towns.keys():
                price = towns[country].get('prices', {}).get(item_type, 0)
                if price > 0:
                    prices.append(price)
            
            if len(prices) > 1:
                # Calculate coefficient of variation (inverse of efficiency)
                cv = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
                efficiency = max(0, 1 - cv)  # Convert to efficiency score
            else:
                efficiency = 1.0  # Perfect efficiency if only one price
            
            efficiency_scores.append(efficiency)
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        bars = ax.bar(item_types, efficiency_scores, color=self.colors[:len(item_types)])
        ax.set_title(title)
        ax.set_xlabel('Item Type')
        ax.set_ylabel('Market Efficiency Score (0-1)')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, efficiency_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def save_plots(self, plots: Dict[str, plt.Figure], directory: str = "plots"):
        """Save all plots to files"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for name, fig in plots.items():
            filename = f"{directory}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved {name} to {filename}")
    
    def create_dashboard(self, towns: Dict[str, Dict], daily_stats: List[Dict],
                        price_data: Dict[str, List[float]], item_types: List[str]):
        """Create a comprehensive dashboard with multiple plots"""
        dashboard_plots = {}
        
        # Price trends
        if price_data:
            dashboard_plots['price_trends'] = self.plot_price_trends(
                price_data, item_types, title="Price Trends Dashboard"
            )
        
        # Price heatmap
        dashboard_plots['price_heatmap'] = self.plot_price_heatmap(
            towns, item_types, title="Price Distribution Heatmap"
        )
        
        # Supply/demand heatmap
        dashboard_plots['supply_demand'] = self.plot_supply_demand_heatmap(
            towns, item_types, title="Supply/Demand Ratio Heatmap"
        )
        
        # Economic indicators
        if daily_stats:
            dashboard_plots['economic_indicators'] = self.plot_economic_indicators(
                daily_stats, title="Economic Indicators Dashboard"
            )
        
        # Market efficiency
        dashboard_plots['market_efficiency'] = self.plot_market_efficiency(
            towns, item_types, title="Market Efficiency Analysis"
        )
        
        return dashboard_plots 
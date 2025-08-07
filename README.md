# Caravan - Quant-Style Economics Simulator

A sophisticated economic simulation that models global trading with dynamic trade routes, price arbitrage, and predictive modeling. The simulator incorporates real-world country relationships, production specializations, and reputation systems to create a realistic trading environment.

## Features

### Core Economic Simulation
- **Dynamic Price Modeling**: Uses regression and predictive modeling for price forecasting
- **Arbitrage Detection**: Identifies profitable trade opportunities across countries
- **Risk Assessment**: Multi-factor risk calculation including political, economic, and reputational factors
- **Market Events**: Dynamic events that affect supply, demand, and prices
- **Trade Network**: Geographic and economic connectivity between countries

### New Features: Reputation System & Production Focus

#### Reputation System
- **Alliance/Enemy Relationships**: Countries have historical relationships that affect trade
- **Dynamic Reputation Updates**: Successful trades improve relationships between countries
- **Risk Impact**: Reputation affects trade risk levels and success rates
- **Political Considerations**: Known hostile relationships increase trade risk

**Reputation Levels:**
- **Ally** (≥0.5): Reduced risk, trade bonuses
- **Friendly** (≥0.2): Slight trade bonuses
- **Neutral** (-0.2 to 0.2): Standard trade conditions
- **Unfriendly** (-0.5 to -0.2): Trade penalties
- **Enemy** (<-0.5): Significant trade penalties and risk

#### Production Focus System
- **Country Specializations**: Each country has production focus multipliers for different item types
- **Efficiency Bonuses**: Higher focus means better production efficiency and supply
- **Trade Optimization**: Production focus affects optimal trade quantities and pricing
- **Economic Realism**: Reflects real-world country specializations

**Example Production Focuses:**
- **USA**: Tech (2.0x), Armaments (2.0x), Medicine (1.5x)
- **China**: Tech (1.8x), Refined Resources (1.8x), Raw Resources (1.5x)
- **Russia**: Raw Resources (2.0x), Energies (2.0x), Armaments (1.5x)
- **Germany**: Tech (1.8x), Refined Resources (1.8x), Armaments (1.5x)
- **Japan**: Tech (2.0x), Delicacies (1.8x), Medicine (1.5x)

### Item Types
- **Food**: Basic sustenance, perishable
- **Raw Resource**: Mining and extraction products
- **Tech**: High-technology products
- **Armaments**: Military and defense equipment
- **Refined Resources**: Processed industrial materials
- **Energies**: Energy products (oil, gas, etc.)
- **Delicacies**: Luxury and specialty goods
- **Medicine**: Healthcare and pharmaceutical products

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Simulation**:
   ```bash
   python main.py --days 30 --visualize
   ```

3.**Outputs**:
   This creates matplotlib files with trendlines as well as a data summary

## Interactive Mode

The simulator includes a comprehensive **interactive mode** where you can play the economic simulation in real-time!

### Getting Started

```bash
python main.py --interactive
```

### Key Features

#### **Real-time Dashboard**
- **Current Status**: Day, simulation status, active caravans, total profit
- **Market Overview**: Live price trends for all items with change indicators
- **Recent Trades**: Last 5 completed trades with routes and values

#### 🎮 **Player Controls**

| Control | Action |
|---------|--------|
| `1` | Step forward 1 day |
| `7` | Step forward 7 days |
| `30`| Step forward 30 days |
| `T` | **Show trade opportunities** (send caravans) |
| `M` | **Manual caravan creation** |
| `C` | Show caravan status |
| `P` | Show price trends |
| `R` | Show reputation matrix |
| `A` | Show arbitrage opportunities |
| `D` | **Show detailed town info** |
| `H` | Show help |
| `Q` | Quit simulation |

### **Trade Opportunities (T)**

**NEW!** Players can now choose which caravans to send:

1. **View Top 10 Opportunities**: See the most profitable trade routes
2. **Automatic Caravan Creation**: Select by number to send caravans
3. **Optimal Quantity Calculation**: System calculates the best trade quantity
4. **Risk Assessment**: See risk levels for each opportunity

**Example:**
```
💰 TOP TRADE OPPORTUNITIES:
------------------------------------------------------------
 1. Tech           | USA         → China       | Profit:  25.3% | Risk:   0.5
 2. Armaments      | Germany     → Russia      | Profit:  18.7% | Risk:   0.8
 3. Medicine       | Switzerland → Brazil      | Profit:  15.2% | Risk:   0.3

🎮 PLAYER ACTIONS:
  Enter opportunity number to send caravan
  Press ENTER to continue without action

Your choice: 1

📦 Sending caravan for Tech
   Route: USA → China
   Quantity: 150
   Expected Profit: 25.3%
✅ Caravan a1b2c3d4... created and dispatched!
```

### 🚚 **Manual Caravan Creation (M)**

1. **View Available Towns**: See numbered list of all countries
2. **Custom Routes**: Choose origin, destination, item type, and quantity
3. **Inventory Check**: System verifies available inventory
4. **Value Calculation**: Shows total cargo value

**Format:** `origin_number,destination_number,item_type,quantity`

**Example:**
```
🚚 MANUAL CARAVAN CREATION
============================================================
📋 AVAILABLE TOWNS:
------------------------------------------------------------
 1. USA            | North America | Pop: 331,002,651
 2. China          | Asia          | Pop: 1,439,323,776
 3. Germany        | Europe        | Pop: 83,783,942
...

🎮 CREATE CARAVAN:
  Format: origin_number,destination_number,item_type,quantity
  Example: 1,5,Tech,100
  Press ENTER to cancel

Your choice: 1,3,Tech,200

✅ Caravan e5f6g7h8... created!
   Route: USA → Germany
   Cargo: 200 Tech
   Value: $113,000.00
```

### 🏘️ **Detailed Town Information (D)**

1. **Town List**: See all available towns with basic info
2. **Detailed View**: Select by number or name for full details
3. **Rich Information**: Prices, inventory, reputation, supply/demand
4. **Visual Indicators**: Symbols for production focus, price trends, supply ratios

**Example:**
```
🏘️  USA - DETAILED INFORMATION
============================================================
📍 BASIC INFO:
  Region: North America
  Population: 331,002,651
  GDP per capita: $63,416
  Tech Level: 9
  Stability: 0.85

📦 PRODUCTION FOCUS:
  ⭐ Tech           :   2.10x
  🔸 Armaments      :   1.45x
  • Food           :   0.95x

🤝 REPUTATION:
  🟢 Allies: UK, Canada, Germany, France, Japan
  🔴 Enemies: Russia, China

💰 CURRENT PRICES:
  ↗️ Tech           : $  565.00 (+12.3%)
  ↘️ Food           : $  104.00 ( -5.2%)
  ➡️ Armaments      : $  312.00 ( +0.1%)

📊 INVENTORY:
  📦 Tech           :    1,250
  📦 Armaments      :      850

📈 SUPPLY/DEMAND:
  🟢 Tech           : Supply  1,250 | Demand  1,000 | Ratio  1.25
  🟡 Armaments      : Supply    850 | Demand    900 | Ratio  0.94
```

### **Key Benefits**

- **Player Agency**: Choose which trades to execute
- **Strategic Decision Making**: Balance profit vs. risk
- **Market Intelligence**: Real-time price and supply data
- **Diplomatic Awareness**: Understand country relationships
- **Economic Mastery**: Learn supply/demand dynamics

### 🎯 **Gameplay Tips**

1. **Start Small**: Begin with low-risk, high-profit opportunities
2. **Monitor Prices**: Watch for price trends and arbitrage opportunities
3. **Check Reputation**: Ally nations offer better trade terms
4. **Balance Supply/Demand**: High supply ratios mean better prices
5. **Diversify Routes**: Don't put all caravans on one route
6. **Watch Inventory**: Ensure towns have enough goods to trade

The interactive mode transforms the economic simulator into a **playable economic strategy game** where you can test your trading skills and market intuition!

## Key Components

### Town Class (Enhanced)
- **Reputation System**: Tracks relationships with other countries
- **Production Focus**: Specialized production capabilities
- **Economic State**: Inventory, prices, supply/demand
- **Geographic Data**: Location for distance calculations

### CaravanDecisionModel (Enhanced)
- **Reputation-Aware Decisions**: Considers country relationships
- **Production Focus Integration**: Accounts for country specializations
- **Multi-Factor Analysis**: Price arbitrage, risk, time, reputation, efficiency
- **Risk Assessment**: Political, economic, and reputational risk factors

### EconomyModel (Enhanced)
- **Reputation-Adjusted Pricing**: Prices reflect country relationships
- **Production Focus Impact**: Supply/demand affected by specializations
- **Enhanced Arbitrage**: Opportunities consider reputation and focus
- **Risk-Adjusted Returns**: Profit calculations include all risk factors

## Decision Model Factors

The caravan decision model evaluates trade opportunities using:

1. **Price Arbitrage** (40% weight): Buy low, sell high opportunities
2. **Travel Time** (20% weight): Distance and route efficiency
3. **Risk Level** (20% weight): Political, economic, and reputational risk
4. **Reputation Impact** (10% weight): Relationship effects on trade
5. **Market Efficiency** (10% weight): Price dispersion and market conditions

### Risk Calculation
- **Base Risk**: 10% minimum risk
- **Distance Risk**: Up to 30% for long routes
- **Stability Risk**: Based on country stability ratings
- **Political Risk**: Known hostile relationships
- **Reputation Risk**: Based on country relationships
- **Item Risk**: Item-specific risk factors
- **Regional Risk**: Cross-regional trade penalties

## Output and Analysis

### Trade Logs
- Timestamp, origin, destination, item type
- Quantity, price per unit, total value
- Caravan ID, risk level, travel time
- Reputation impact and production focus bonuses

### Visualizations
- Price trends and volatility
- Supply/demand heatmaps
- Trade network connectivity
- Economic indicators
- Market efficiency analysis

### Statistics
- Global GDP and trade volume
- Country-specific statistics
- Market analysis and trends
- Risk-adjusted performance metrics

## Customization

### Adding New Countries
1. Add country data to `data/towns.csv`
2. Define reputation relationships in `Town._initialize_reputation()`
3. Set production focus in `Town._initialize_production_focus()`

### Modifying Reputation System
- Update alliance/enemy lists in `Town._initialize_reputation()`
- Adjust reputation change rates in `EconomicSimulator._update_reputation_from_trade()`
- Modify risk calculations in `CaravanDecisionModel._calculate_reputation_risk()`

### Adjusting Production Focus
- Modify focus multipliers in `Town._initialize_production_focus()`
- Update impact calculations in `EconomyModel._calculate_production_focus_impact_for_arbitrage()`
- Adjust quantity calculations in `EconomyModel.calculate_optimal_trade_quantity()`

## Future Enhancements

- **Dynamic Alliances**: Reputation changes affecting alliance formation
- **Trade Agreements**: Formal trade partnerships with bonuses
- **Economic Sanctions**: Political restrictions on trade
- **Currency Exchange**: Multi-currency trading system
- **Advanced AI**: Machine learning for trade route optimization
- **Real-time Data**: Integration with real-world economic data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

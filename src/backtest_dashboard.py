"""
BACKTEST DASHBOARD WITH MONTE CARLO - V4.0
FINAL VERSION - Works with NEW CSV structure from modified backtester
- Handles 'date' as regular column (not index)
- Uses correct column names from new CSV format
- All metrics calculated and displayed
- Production ready
"""

from flask import Flask, render_template_string, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import traceback
import warnings
from GetFreshMarketData import *
import webbrowser
import threading
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==================== CONFIGURATION ====================
# UPDATE THESE PATHS TO YOUR DATA LOCATION
MC_FILE = TEMP/"monte_carlo_results.pkl"
# TEMP = Path("./outputs")


def load_pickle_data():
    """Load Monte Carlo pickle file with detailed error reporting"""
    try:
        with open(MC_FILE, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ MC File loaded: {MC_FILE}")
        print(f"   Keys: {list(data.keys())}")
        
        # Check what we have
        if 'cagr_distribution' in data:
            ca = np.array(data['cagr_distribution'])
            print(f"   cagr_distribution: {len(ca)} items, range=[{ca.min():.4f}, {ca.max():.4f}]")
        
        if 'max_dd_distribution' in data:
            dd = np.array(data['max_dd_distribution'])
            print(f"   max_dd_distribution: {len(dd)} items, range=[{dd.min():.4f}, {dd.max():.4f}]")
        
        if 'sample_curves' in data:
            curves = data['sample_curves']
            print(f"   sample_curves: {len(curves)} curves, first curve length={len(curves[0]) if len(curves) > 0 else 0}")
        
        if 'stats' in data:
            stats = data['stats']
            print(f"   stats keys: {list(stats.keys()) if isinstance(stats, dict) else 'not a dict'}")
        
        return data
    except Exception as e:
        print(f"❌ Error loading pickle: {e}")
        traceback.print_exc()
        return None

def load_csv(filename):
    """Load CSV file from outputs folder"""
    try:
        file_path = TEMP / filename
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        return None

def extract_final_equity_from_curves(sample_curves):
    """V4.4 FIX: Extract final equity values from equity curves"""
    try:
        final_equity_list = []
        for curve in sample_curves:
            if len(curve) > 0:
                # Last value in each curve is the final equity
                final_equity_list.append(curve[-1])
        return final_equity_list
    except Exception as e:
        print(f"❌ Error extracting final equity: {e}")
        return []

# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/metrics')
def get_metrics():
    """Calculate all backtest performance metrics"""
    try:
        equity = load_csv('equity_curve.csv')
        trades = load_csv('trades.csv')
        
        if equity is None or equity.empty:
            return jsonify({"error": "No equity curve data"}), 404
        
        # Structure: [index], date, equity
        # Drop the index column if present
        if equity.columns[0] in ['', 'Unnamed: 0', 'index']:
            equity = equity.iloc[:, 1:]
        
        # Now equity has: date, equity
        date_col = 'date'
        equity_col = 'equity'
        
        # Ensure columns exist
        if date_col not in equity.columns or equity_col not in equity.columns:
            return jsonify({"error": f"Missing columns. Found: {equity.columns.tolist()}"}), 404
        
        # Convert date to datetime
        equity[date_col] = pd.to_datetime(equity[date_col])
        equity = equity.sort_values(date_col).reset_index(drop=True)
        
        # Basic stats
        start_date = equity[date_col].iloc[0]
        end_date = equity[date_col].iloc[-1]
        initial_capital = equity[equity_col].iloc[0]
        final_equity = equity[equity_col].iloc[-1]
        
        # Duration in years
        days = (end_date - start_date).days
        years = days / 365.25
        
        # CAGR
        cagr = ((final_equity / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Max Drawdown Percentage
        running_max = equity[equity_col].expanding().max()
        drawdown_pct = (equity[equity_col] - running_max) / running_max * 100
        max_dd_pct = drawdown_pct.min()
        
        # Max Drawdown Duration (in days)
        max_dd_idx = drawdown_pct.idxmin()
        running_max_at_dd = running_max.iloc[max_dd_idx]
        
        # Find recovery point
        recovery_idx = len(equity) - 1
        for i in range(max_dd_idx + 1, len(equity)):
            if equity[equity_col].iloc[i] >= running_max_at_dd:
                recovery_idx = i
                break
        
        max_dd_duration = (equity[date_col].iloc[recovery_idx] - equity[date_col].iloc[max_dd_idx]).days
        
        # Trade statistics
        num_trades = len(trades) if trades is not None and not trades.empty else 0
        win_rate = 0.0
        avg_trade_pnl = 0.0
        car_over_mdd = 0.0
        sharpe = 0.0
        sortino = 0.0
        
        if num_trades > 0 and trades is not None:
            trades = trades.dropna(subset=['profit_loss_amount'])
            if len(trades) > 0:
                # Win rate
                winning_trades = len(trades[trades['profit_loss_amount'] > 0])
                win_rate = (winning_trades / len(trades)) * 100
                
                # Average trade PnL
                avg_trade_pnl = trades['profit_loss_amount'].mean()
                
                # CAR/MDD ratio
                car_over_mdd = cagr / abs(max_dd_pct) if max_dd_pct != 0 else 0
                
                # Sharpe Ratio (annualized)
                returns = equity[equity_col].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
                
                # Sortino Ratio (annualized, only negative returns)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    if downside_std > 0:
                        sortino = (returns.mean() / downside_std) * np.sqrt(252)
        
        # Format metrics in CORRECT ORDER
        metrics = {
            'backtest_start_date': start_date.strftime('%Y-%m-%d'),
            'backtest_end_date': end_date.strftime('%Y-%m-%d'),
            'duration_years': f"{years:.2f}",
            'initial_equity': f"₹{initial_capital:,.0f}",
            'final_equity': f"₹{final_equity:,.0f}",
            'cagr': f"{cagr:.2f}%",
            'max_dd': f"{max_dd_pct:.2f}%",
            'max_dd_duration_days': f"{int(max_dd_duration)}",
            'number_of_trades': f"{int(num_trades)}",
            'win_rate': f"{win_rate:.2f}%",
            'car_over_mdd': f"{car_over_mdd:.2f}",
            'sharpe': f"{sharpe:.2f}",
            'sortino': f"{sortino:.2f}",
            'avg_trade_pnl': f"₹{avg_trade_pnl:,.0f}",
        }
        
        return jsonify(metrics)
    
    except Exception as e:
        print(f"❌ Error in get_metrics: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/equity-curve')
def get_equity_curve():
    """Get equity curve data for chart"""
    try:
        equity = load_csv('equity_curve.csv')
        
        if equity is None or equity.empty:
            return jsonify({"dates": [], "values": []}), 200
        
        # Drop index column
        if equity.columns[0] in ['', 'Unnamed: 0', 'index']:
            equity = equity.iloc[:, 1:]
        
        date_col = 'date'
        equity_col = 'equity'
        
        equity[date_col] = pd.to_datetime(equity[date_col])
        equity = equity.sort_values(date_col)
        
        return jsonify({
            'dates': equity[date_col].dt.strftime('%Y-%m-%d').tolist(),
            'values': equity[equity_col].tolist()
        })
    except Exception as e:
        print(f"Error in equity-curve: {e}")
        return jsonify({"dates": [], "values": []}), 200

@app.route('/api/drawdown')
def get_drawdown():
    """Get drawdown data - recalculate from equity curve"""
    try:
        equity = load_csv('equity_curve.csv')
        
        if equity is None or equity.empty:
            return jsonify({"dates": [], "values": []}), 200
        
        # Drop index column
        if equity.columns[0] in ['', 'Unnamed: 0', 'index']:
            equity = equity.iloc[:, 1:]
        
        date_col = 'date'
        equity_col = 'equity'
        
        equity[date_col] = pd.to_datetime(equity[date_col])
        equity = equity.sort_values(date_col)
        
        # Calculate drawdown from equity curve
        running_max = equity[equity_col].expanding().max()
        drawdown = (equity[equity_col] - running_max) / running_max * 100
        
        return jsonify({
            'dates': equity[date_col].dt.strftime('%Y-%m-%d').tolist(),
            'values': drawdown.tolist()
        })
    except Exception as e:
        print(f"Error in drawdown: {e}")
        return jsonify({"dates": [], "values": []}), 200

@app.route('/api/trade-analysis')
def get_trade_analysis():
    """Trade P/L distribution histogram"""
    try:
        trades = load_csv('trades.csv')
        
        if trades is None or len(trades) == 0:
            return jsonify({"bins": [], "counts": []}), 200
        
        # Get profit_loss_amount column
        if 'profit_loss_amount' not in trades.columns:
            return jsonify({"bins": [], "counts": []}), 200
        
        pnl = trades['profit_loss_amount'].dropna()
        
        if len(pnl) == 0:
            return jsonify({"bins": [], "counts": []}), 200
        
        # Create histogram
        counts, edges = np.histogram(pnl, bins=20)
        bin_centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
        
        return jsonify({
            'bins': bin_centers,
            'counts': counts.tolist()
        })
    except Exception as e:
        print(f"Error in trade-analysis: {e}")
        return jsonify({"bins": [], "counts": []}), 200

@app.route('/api/symbol-stats')
def get_symbol_stats():
    """Symbol performance statistics"""
    try:
        trades = load_csv('trades.csv')
        
        if trades is None or len(trades) == 0:
            return jsonify([]), 200
        
        if 'symbol' not in trades.columns or 'profit_loss_amount' not in trades.columns:
            return jsonify([]), 200
        
        # Group by symbol
        symbol_groups = trades.groupby('symbol').agg({
            'profit_loss_amount': ['sum', 'count', 'mean']
        }).round(2)
        
        result = []
        for symbol in symbol_groups.index:
            total_pnl = symbol_groups.loc[symbol, ('profit_loss_amount', 'sum')]
            count = int(symbol_groups.loc[symbol, ('profit_loss_amount', 'count')])
            mean_pnl = symbol_groups.loc[symbol, ('profit_loss_amount', 'mean')]
            
            # Calculate win rate for this symbol
            symbol_trades = trades[trades['symbol'] == symbol]
            wins = len(symbol_trades[symbol_trades['profit_loss_amount'] > 0])
            wr = (wins / count * 100) if count > 0 else 0
            
            result.append({
                'symbol': symbol,
                'total_pnl': f"₹{total_pnl:,.0f}",
                'trades': count,
                'avg_pnl': f"₹{mean_pnl:,.0f}",
                'win_rate': f"{wr:.2f}%"
            })
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in symbol-stats: {e}")
        return jsonify([]), 200

# ==================== MONTE CARLO ROUTES ====================

@app.route('/api/mc-summary')
def get_mc_summary():
    """Monte Carlo summary metrics in CORRECT ORDER"""
    try:
        mc_data = load_pickle_data()
        
        if mc_data is None:
            return jsonify({"error": "No MC data"}), 404
        
        cagr_list = np.array(mc_data.get('cagr_distribution', []))
        max_dd_list = np.array(mc_data.get('max_dd_distribution', []))
        
        if len(cagr_list) == 0 or len(max_dd_list) == 0:
            return jsonify({"error": "Empty MC data"}), 404
        
        # Convert to percentages if needed (check the actual range)
        if cagr_list.max() < 1:  # Values like 0.10 (10%)
            cagr_list = cagr_list * 100
        
        # For max_dd: values are negative like -0.20 or -20 depending on format
        if abs(max_dd_list.max()) < 1:  # Values like -0.20 (-20%)
            max_dd_list = max_dd_list * 100
        
        # ORDER: mean_cagr, median_cagr, worst_5_cagr, mean_max_dd, median_max_dd, worst_5_max_dd
        summary = {
            'mean_cagr': f"{cagr_list.mean():.2f}%",
            'median_cagr': f"{np.median(cagr_list):.2f}%",
            'worst_5_cagr': f"{np.percentile(cagr_list, 5):.2f}%",
            'mean_max_dd': f"{max_dd_list.mean():.2f}%",
            'median_max_dd': f"{np.median(max_dd_list):.2f}%",
            'worst_5_max_dd': f"{np.percentile(max_dd_list, 5):.2f}%"
        }
        
        return jsonify(summary)
    except Exception as e:
        print(f"Error in mc-summary: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/mc-percentiles')
def get_mc_percentiles():
    """V4.4: Extract final_equity from sample_curves last values"""
    try:
        mc_data = load_pickle_data()
        
        if mc_data is None:
            print("❌ MC data is None")
            return jsonify([]), 404
        
        # Extract arrays
        cagr_list = mc_data.get('cagr_distribution')
        max_dd_list = mc_data.get('max_dd_distribution')
        sample_curves = mc_data.get('sample_curves')
        
        # V4.4 FIX: Extract final_equity from sample_curves
        final_equity_list = extract_final_equity_from_curves(sample_curves) if sample_curves else []
        
        # Debug logging
        print(f"\n🔍 DEBUG: Percentile Analysis")
        print(f"   final_equity_list (from curves): len={len(final_equity_list)}")
        print(f"   cagr_list type: {type(cagr_list)}, len={len(cagr_list) if cagr_list is not None else 'None'}")
        print(f"   max_dd_list type: {type(max_dd_list)}, len={len(max_dd_list) if max_dd_list is not None else 'None'}")
        
        if len(final_equity_list) == 0 or cagr_list is None or max_dd_list is None:
            print("❌ Missing required data for percentile analysis")
            print(f"   final_equity count: {len(final_equity_list)}")
            print(f"   cagr_list: {cagr_list is not None}")
            print(f"   max_dd_list: {max_dd_list is not None}")
            return jsonify([]), 404
        
        # Convert to numpy arrays
        final_equity_list = np.array(final_equity_list)
        cagr_list = np.array(cagr_list)
        max_dd_list = np.array(max_dd_list)
        
        print(f"   final_equity range: [{final_equity_list.min():,.0f}, {final_equity_list.max():,.0f}]")
        print(f"   cagr range: [{cagr_list.min():.4f}, {cagr_list.max():.4f}]")
        print(f"   max_dd range: [{max_dd_list.min():.4f}, {max_dd_list.max():.4f}]")
        
        # Convert to percentages if needed
        if cagr_list.max() < 1 and cagr_list.max() > 0:  # e.g., 0.10 means 10%
            print("   Converting CAGR from decimal to percentage")
            cagr_list = cagr_list * 100
        
        if abs(max_dd_list.max()) < 1:  # e.g., -0.20 means -20%
            print("   Converting Max DD from decimal to percentage")
            max_dd_list = max_dd_list * 100
        
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        result = []
        
        for p in percentiles:
            final_eq = np.percentile(final_equity_list, p)
            cagr = np.percentile(cagr_list, p)
            max_dd = np.percentile(max_dd_list, p)
            
            result.append({
                'percentile': f"{p}%",
                'final_equity': f"₹{final_eq:,.0f}",
                'cagr': f"{cagr:.2f}%",
                'max_dd': f"{max_dd:.2f}%"
            })
        
        print(f"✅ Percentile Analysis: {len(result)} rows created")
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in mc-percentiles: {e}")
        traceback.print_exc()
        return jsonify([]), 200

@app.route('/api/mc-max-dd-histogram')
def get_mc_max_dd_histogram():
    """Max DD distribution histogram"""
    try:
        mc_data = load_pickle_data()
        
        if mc_data is None:
            return jsonify({"bins": [], "counts": []}), 404
        
        max_dd_list = np.array(mc_data.get('max_dd_distribution', []))
        
        if len(max_dd_list) == 0:
            return jsonify({"bins": [], "counts": []}), 404
        
        if abs(max_dd_list.max()) < 1:
            max_dd_list = max_dd_list * 100
        
        counts, edges = np.histogram(max_dd_list, bins=20)
        bin_centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
        
        return jsonify({
            'bins': bin_centers,
            'counts': counts.tolist()
        })
    except Exception as e:
        print(f"Error in mc-max-dd-histogram: {e}")
        return jsonify({"bins": [], "counts": []}), 200

@app.route('/api/mc-cagr-histogram')
def get_mc_cagr_histogram():
    """CAGR distribution histogram"""
    try:
        mc_data = load_pickle_data()
        
        if mc_data is None:
            return jsonify({"bins": [], "counts": []}), 404
        
        cagr_list = np.array(mc_data.get('cagr_distribution', []))
        
        if len(cagr_list) == 0:
            return jsonify({"bins": [], "counts": []}), 404
        
        if cagr_list.max() < 1 and cagr_list.max() > 0:
            cagr_list = cagr_list * 100
        
        counts, edges = np.histogram(cagr_list, bins=20)
        bin_centers = [(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)]
        
        return jsonify({
            'bins': bin_centers,
            'counts': counts.tolist()
        })
    except Exception as e:
        print(f"Error in mc-cagr-histogram: {e}")
        return jsonify({"bins": [], "counts": []}), 200

@app.route('/api/mc-scatter')
def get_mc_scatter():
    """Max DD vs CAGR scatter plot"""
    try:
        mc_data = load_pickle_data()
        
        if mc_data is None:
            return jsonify({"x": [], "y": []}), 404
        
        max_dd_list = np.array(mc_data.get('max_dd_distribution', []))
        cagr_list = np.array(mc_data.get('cagr_distribution', []))
        
        if len(max_dd_list) == 0 or len(cagr_list) == 0:
            return jsonify({"x": [], "y": []}), 404
        
        if cagr_list.max() < 1 and cagr_list.max() > 0:
            cagr_list = cagr_list * 100
        if abs(max_dd_list.max()) < 1:
            max_dd_list = max_dd_list * 100
        
        return jsonify({
            'x': max_dd_list.tolist(),
            'y': cagr_list.tolist()
        })
    except Exception as e:
        print(f"Error in mc-scatter: {e}")
        return jsonify({"x": [], "y": []}), 200

# ==================== HTML TEMPLATE ====================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Dashboard with Monte Carlo - V4.4</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 { font-size: 2.5em; margin-bottom: 5px; }
        .header p { font-size: 1.1em; opacity: 0.9; }
        
        .container { max-width: 1600px; margin: 0 auto; }
        
        .section-title {
            font-size: 1.8em;
            color: white;
            margin: 40px 0 20px 0;
            padding-bottom: 10px;
            border-bottom: 3px solid white;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: white;
            padding: 18px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .metric-card .label {
            font-size: 0.75em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        .metric-card .value {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            word-break: break-word;
        }
        
        .metric-card.positive .value { color: #27ae60; }
        .metric-card.negative .value { color: #e74c3c; }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            position: relative;
        }
        
        .chart-title {
            font-size: 1.1em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        
        canvas { max-height: 300px; }
        
        .table-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        table thead {
            background: #667eea;
            color: white;
        }
        
        table th, table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        table tbody tr:hover { background: #f5f5f5; }
        
        .loading {
            text-align: center;
            color: white;
            font-size: 1.1em;
        }
        
        @media (max-width: 1200px) {
            .charts-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎲 Backtest Dashboard with Monte Carlo</h1>
        <p>V4.4</p>
    </div>
    
    <div class="container">
        
        <!-- BACKTEST SECTION -->
        <div class="section-title">📊 Backtest Performance Metrics</div>
        
        <div class="metrics-grid" id="backtest-metrics">
            <div class="loading">Loading metrics...</div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Equity Curve</div>
                <canvas id="equity-curve-chart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Drawdown</div>
                <canvas id="drawdown-chart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Trade P/L Distribution</div>
                <canvas id="trade-distribution-chart"></canvas>
            </div>
        </div>
        
        <div class="table-container">
            <div class="chart-title">Symbol Performance</div>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Total P/L</th>
                        <th>Trades</th>
                        <th>Avg P/L</th>
                        <th>Win Rate</th>
                    </tr>
                </thead>
                <tbody id="symbol-stats"></tbody>
            </table>
        </div>
        
        <!-- MONTE CARLO SECTION -->
        <div class="section-title">🎲 Monte Carlo Analysis</div>
        
        <div class="metrics-grid" id="mc-metrics">
            <div class="loading">Loading Monte Carlo data...</div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Max Drawdown Distribution</div>
                <canvas id="max-dd-histogram-chart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">CAGR Distribution</div>
                <canvas id="cagr-histogram-chart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Max DD vs CAGR</div>
                <canvas id="scatter-chart"></canvas>
            </div>
        </div>
        
        <div class="table-container">
            <div class="chart-title">Percentile Analysis</div>
            <table>
                <thead>
                    <tr>
                        <th>Percentile</th>
                        <th>Final Equity</th>
                        <th>CAGR %</th>
                        <th>Max DD %</th>
                    </tr>
                </thead>
                <tbody id="mc-percentiles"></tbody>
            </table>
        </div>
        
    </div>
    
    <script>
        let charts = {};
        
        function loadBacktestMetrics() {
            fetch('/api/metrics')
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('backtest-metrics').innerHTML = '<div class="loading" style="grid-column: 1/-1;">Error: ' + data.error + '</div>';
                        return;
                    }
                    let html = '';
                    const order = ['backtest_start_date', 'backtest_end_date', 'duration_years', 'initial_equity', 'final_equity', 'cagr', 'max_dd', 'max_dd_duration_days', 'number_of_trades', 'win_rate', 'car_over_mdd', 'sharpe', 'sortino', 'avg_trade_pnl'];
                    
                    order.forEach(key => {
                        if (key in data) {
                            let value = data[key];
                            let className = 'metric-card';
                            if (value.includes('-') && !key.includes('date')) className += ' negative';
                            if ((value.includes('%') || value.includes('₹')) && !value.includes('-') && !key.includes('date')) className += ' positive';
                            
                            html += `
                                <div class="${className}">
                                    <div class="label">${key.replace(/_/g, ' ')}</div>
                                    <div class="value">${value}</div>
                                </div>
                            `;
                        }
                    });
                    document.getElementById('backtest-metrics').innerHTML = html || '<div class="loading">No data</div>';
                })
                .catch(e => console.error('Error:', e));
        }
        
        function loadEquityCurve() {
            fetch('/api/equity-curve')
                .then(r => r.json())
                .then(data => {
                    if (!data.dates || data.dates.length === 0) return;
                    const ctx = document.getElementById('equity-curve-chart');
                    if (charts['equity']) charts['equity'].destroy();
                    charts['equity'] = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [{
                                label: 'Equity',
                                data: data.values,
                                borderColor: '#667eea',
                                fill: true,
                                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                                tension: 0.1,
                                pointRadius: 0
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: { legend: { display: false } }
                        }
                    });
                })
                .catch(e => console.error('Error loading equity:', e));
        }
        
        function loadDrawdown() {
            fetch('/api/drawdown')
                .then(r => r.json())
                .then(data => {
                    if (!data.dates || data.dates.length === 0) return;
                    const ctx = document.getElementById('drawdown-chart');
                    if (charts['drawdown']) charts['drawdown'].destroy();
                    charts['drawdown'] = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [{
                                label: 'Drawdown %',
                                data: data.values,
                                borderColor: '#e74c3c',
                                fill: true,
                                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                                tension: 0.1,
                                pointRadius: 0
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: { legend: { display: false } }
                        }
                    });
                })
                .catch(e => console.error('Error loading drawdown:', e));
        }
        
        function loadTradeAnalysis() {
            fetch('/api/trade-analysis')
                .then(r => r.json())
                .then(data => {
                    if (!data.bins || data.bins.length === 0) return;
                    const ctx = document.getElementById('trade-distribution-chart');
                    if (charts['trades']) charts['trades'].destroy();
                    charts['trades'] = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.bins.map(b => b.toFixed(0)),
                            datasets: [{
                                label: 'Trade Count',
                                data: data.counts,
                                backgroundColor: '#f39c12'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: { legend: { display: false } }
                        }
                    });
                })
                .catch(e => console.error('Error loading trades:', e));
        }
        
        function loadSymbolStats() {
            fetch('/api/symbol-stats')
                .then(r => r.json())
                .then(data => {
                    let html = '';
                    if (Array.isArray(data) && data.length > 0) {
                        data.forEach(row => {
                            html += `<tr><td>${row.symbol}</td><td>${row.total_pnl}</td><td>${row.trades}</td><td>${row.avg_pnl}</td><td>${row.win_rate}</td></tr>`;
                        });
                    }
                    document.getElementById('symbol-stats').innerHTML = html || '<tr><td colspan="5" style="text-align:center;">No data</td></tr>';
                })
                .catch(e => console.error('Error:', e));
        }
        
        function loadMCMetrics() {
            fetch('/api/mc-summary')
                .then(r => r.json())
                .then(data => {
                    if (data.error) {
                        console.error('MC Summary error:', data.error);
                        return;
                    }
                    let html = '';
                    const order = ['mean_cagr', 'median_cagr', 'worst_5_cagr', 'mean_max_dd', 'median_max_dd', 'worst_5_max_dd'];
                    order.forEach(key => {
                        if (key in data) {
                            let className = 'metric-card';
                            if (key.includes('dd')) className += ' negative';
                            if (key.includes('cagr')) className += ' positive';
                            html += `<div class="${className}"><div class="label">${key.replace(/_/g, ' ')}</div><div class="value">${data[key]}</div></div>`;
                        }
                    });
                    document.getElementById('mc-metrics').innerHTML = html || '<div class="loading">No MC metrics available</div>';
                })
                .catch(e => {
                    console.error('Error loading MC metrics:', e);
                    document.getElementById('mc-metrics').innerHTML = '<div class="loading">Failed to load MC metrics</div>';
                });
        }
        
        function loadMCPercentiles() {
            fetch('/api/mc-percentiles')
                .then(r => {
                    console.log('Percentile response status:', r.status);
                    return r.json();
                })
                .then(data => {
                    console.log('Percentile data received:', data);
                    let html = '';
                    if (Array.isArray(data) && data.length > 0) {
                        console.log('Creating percentile rows:', data.length);
                        data.forEach((row, idx) => {
                            console.log(`Row ${idx}:`, row);
                            html += `<tr><td>${row.percentile}</td><td>${row.final_equity}</td><td>${row.cagr}</td><td>${row.max_dd}</td></tr>`;
                        });
                    } else {
                        console.log('No percentile data or not an array');
                        html = '<tr><td colspan="4" style="text-align:center;">No percentile data available</td></tr>';
                    }
                    document.getElementById('mc-percentiles').innerHTML = html;
                })
                .catch(e => {
                    console.error('Error loading percentiles:', e);
                    document.getElementById('mc-percentiles').innerHTML = '<tr><td colspan="4" style="text-align:center;">Error loading percentile data</td></tr>';
                });
        }
        
        function loadMaxDDHistogram() {
            fetch('/api/mc-max-dd-histogram')
                .then(r => r.json())
                .then(data => {
                    if (!data.bins || data.bins.length === 0) return;
                    const ctx = document.getElementById('max-dd-histogram-chart');
                    if (charts['dd-hist']) charts['dd-hist'].destroy();
                    charts['dd-hist'] = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.bins.map(b => b.toFixed(1)),
                            datasets: [{
                                label: 'Frequency',
                                data: data.counts,
                                backgroundColor: '#e74c3c'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: { legend: { display: false } }
                        }
                    });
                })
                .catch(e => console.error('Error:', e));
        }
        
        function loadCAGRHistogram() {
            fetch('/api/mc-cagr-histogram')
                .then(r => r.json())
                .then(data => {
                    if (!data.bins || data.bins.length === 0) return;
                    const ctx = document.getElementById('cagr-histogram-chart');
                    if (charts['cagr-hist']) charts['cagr-hist'].destroy();
                    charts['cagr-hist'] = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: data.bins.map(b => b.toFixed(1)),
                            datasets: [{
                                label: 'Frequency',
                                data: data.counts,
                                backgroundColor: '#27ae60'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            plugins: { legend: { display: false } }
                        }
                    });
                })
                .catch(e => console.error('Error:', e));
        }
        
        function loadScatter() {
            fetch('/api/mc-scatter')
                .then(r => r.json())
                .then(data => {
                    if (!data.x || data.x.length === 0) return;
                    const ctx = document.getElementById('scatter-chart');
                    if (charts['scatter']) charts['scatter'].destroy();
                    const scatterData = data.x.map((x, i) => ({x, y: data.y[i]}));
                    charts['scatter'] = new Chart(ctx, {
                        type: 'scatter',
                        data: {
                            datasets: [{
                                label: 'Simulations',
                                data: scatterData,
                                backgroundColor: 'rgba(102, 126, 234, 0.6)'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: true,
                            scales: {
                                x: { title: { display: true, text: 'Max DD %' } },
                                y: { title: { display: true, text: 'CAGR %' } }
                            }
                        }
                    });
                })
                .catch(e => console.error('Error:', e));
        }
        
        function refreshAll() {
            console.log('Refreshing all data...');
            loadBacktestMetrics();
            loadEquityCurve();
            loadDrawdown();
            loadTradeAnalysis();
            loadSymbolStats();
            loadMCMetrics();
            loadMCPercentiles();
            loadMaxDDHistogram();
            loadCAGRHistogram();
            loadScatter();
        }
        
        console.log('Page loaded, refreshing data...');
        refreshAll();
        setInterval(refreshAll, 30000);
    </script>
</body>
</html>
"""

# ==================== RUN ====================

def open_browser(port=5000):
    """Open browser after server starts"""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{port}')

def display_results():
    print("\n" + "="*70)
    print("🚀 BACKTEST DASHBOARD WITH MONTE CARLO V4.4")
    print("="*70)
    print(f"📁 Data location: {TEMP}")
    print(f"🎲 Monte Carlo file: {MC_FILE}")
    print(f"🌐 Dashboard: http://localhost:5000")
    print(f"💱 Currency: INR (₹)")
    print(f"📋 CSV Format: date (column), equity (column)")
    print(f"⏹️  Stop with: CTRL+C")
    print("="*70)
    print("✅ Fixes:")
    print("   1. Percentile Analysis: Extracts final_equity from sample_curves[-1]")
    print("   2. MC Data Structure: Works with actual prepare_monte_carlo_data() output")
    print("   3. Final Equity: Calculated from last value of each simulated equity curve")
    print("="*70 + "\n")

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    try:
        app.run(debug=True, port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped.")


if __name__ == '__main__':
    display_results()
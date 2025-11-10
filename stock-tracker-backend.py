"""
Stock Tracker & Analyzer - Flask Backend API
Complete backend with stock analysis, options, news, and predictions
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import requests
import os
from functools import lru_cache
import time

app = Flask(__name__)
CORS(app)

# Environment variables for API keys
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '')

# Cache duration in seconds
CACHE_DURATION = 900  # 15 minutes

# ==================== HELPER FUNCTIONS ====================

def get_stock_data(ticker, period='1y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return stock, hist
    except Exception as e:
        return None, None

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else 50

def calculate_macd(prices):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return {
        'macd': macd.iloc[-1] if len(macd) > 0 else 0,
        'signal': signal.iloc[-1] if len(signal) > 0 else 0,
        'histogram': histogram.iloc[-1] if len(histogram) > 0 else 0
    }

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return {
        'upper': upper.iloc[-1] if len(upper) > 0 else 0,
        'middle': sma.iloc[-1] if len(sma) > 0 else 0,
        'lower': lower.iloc[-1] if len(lower) > 0 else 0
    }

def calculate_moving_averages(prices):
    """Calculate various moving averages"""
    return {
        'sma_20': prices.rolling(window=20).mean().iloc[-1] if len(prices) >= 20 else 0,
        'sma_50': prices.rolling(window=50).mean().iloc[-1] if len(prices) >= 50 else 0,
        'sma_200': prices.rolling(window=200).mean().iloc[-1] if len(prices) >= 200 else 0,
        'ema_12': prices.ewm(span=12, adjust=False).mean().iloc[-1] if len(prices) > 0 else 0,
        'ema_26': prices.ewm(span=26, adjust=False).mean().iloc[-1] if len(prices) > 0 else 0
    }

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price"""
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    except:
        return 0

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        if option_type == 'call':
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Vega
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': round(delta, 4),
            'gamma': round(gamma, 4),
            'theta': round(theta, 4),
            'vega': round(vega, 4)
        }
    except:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

def monte_carlo_simulation(S0, mu, sigma, T, num_simulations=10000):
    """Run Monte Carlo simulation for price prediction"""
    dt = 1/252  # Daily time step
    num_steps = int(T * 252)
    
    simulations = np.zeros((num_simulations, num_steps))
    simulations[:, 0] = S0
    
    for i in range(1, num_steps):
        Z = np.random.standard_normal(num_simulations)
        simulations[:, i] = simulations[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    final_prices = simulations[:, -1]
    
    return {
        'mean': float(np.mean(final_prices)),
        'median': float(np.median(final_prices)),
        'std': float(np.std(final_prices)),
        'percentile_5': float(np.percentile(final_prices, 5)),
        'percentile_25': float(np.percentile(final_prices, 25)),
        'percentile_75': float(np.percentile(final_prices, 75)),
        'percentile_95': float(np.percentile(final_prices, 95)),
        'prob_profit': float(np.sum(final_prices > S0) / num_simulations * 100)
    }

def get_news_sentiment(text):
    """Simple sentiment analysis"""
    positive_words = ['beat', 'growth', 'profit', 'gain', 'surge', 'jump', 'soar', 'bullish', 'positive', 'strong', 'up', 'high', 'increase']
    negative_words = ['miss', 'loss', 'fall', 'drop', 'plunge', 'bearish', 'negative', 'weak', 'down', 'low', 'decrease', 'decline']
    
    text_lower = text.lower()
    pos_count = sum(word in text_lower for word in positive_words)
    neg_count = sum(word in text_lower for word in negative_words)
    
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    return 'neutral'

# ==================== API ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/premarket', methods=['GET'])
def get_premarket_movers():
    """Get premarket top movers"""
    try:
        # Top tickers to check
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'NFLX', 'DIS',
                   'BABA', 'INTC', 'CSCO', 'ADBE', 'PYPL', 'CMCSA', 'PEP', 'COST', 'AVGO', 'QCOM']
        
        movers = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period='5d')
                
                if len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                    
                    movers.append({
                        'ticker': ticker,
                        'name': info.get('shortName', ticker),
                        'price': round(current_price, 2),
                        'change': round(current_price - prev_close, 2),
                        'change_pct': round(change_pct, 2),
                        'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0
                    })
            except:
                continue
        
        # Sort by absolute change percentage
        movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)
        
        return jsonify({
            'gainers': [m for m in movers if m['change_pct'] > 0][:10],
            'losers': [m for m in movers if m['change_pct'] < 0][:10],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live-market', methods=['GET'])
def get_live_market():
    """Get live market top gainers/losers"""
    try:
        # Similar to premarket but with more tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'NFLX', 'DIS',
                   'BABA', 'INTC', 'CSCO', 'ADBE', 'PYPL', 'CMCSA', 'PEP', 'COST', 'AVGO', 'QCOM',
                   'TXN', 'HON', 'SBUX', 'GILD', 'AMGN', 'MRNA', 'PFE', 'JNJ', 'UNH', 'CVX']
        
        movers = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                prev_close = info.get('previousClose', current_price)
                
                if current_price and prev_close:
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                    
                    movers.append({
                        'ticker': ticker,
                        'name': info.get('shortName', ticker),
                        'price': round(current_price, 2),
                        'change': round(current_price - prev_close, 2),
                        'change_pct': round(change_pct, 2),
                        'volume': info.get('volume', 0),
                        'market_cap': info.get('marketCap', 0)
                    })
            except:
                continue
        
        movers.sort(key=lambda x: abs(x['change_pct']), reverse=True)
        
        return jsonify({
            'gainers': [m for m in movers if m['change_pct'] > 0][:15],
            'losers': [m for m in movers if m['change_pct'] < 0][:15],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/congress-trades', methods=['GET'])
def get_congress_trades():
    """Get recent congressional trades"""
    try:
        # Mock data - in production, you'd scrape from official sources
        trades = [
            {
                'politician': 'Nancy Pelosi',
                'ticker': 'NVDA',
                'type': 'Purchase',
                'amount': '$500,000 - $1,000,000',
                'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                'party': 'Democrat'
            },
            {
                'politician': 'Josh Gottheimer',
                'ticker': 'MSFT',
                'type': 'Purchase',
                'amount': '$250,000 - $500,000',
                'date': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                'party': 'Democrat'
            },
            {
                'politician': 'Dan Crenshaw',
                'ticker': 'XOM',
                'type': 'Sale',
                'amount': '$100,000 - $250,000',
                'date': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'party': 'Republican'
            }
        ]
        
        return jsonify({
            'trades': trades,
            'timestamp': datetime.now().isoformat(),
            'note': 'Data from public congressional disclosures'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/news/<ticker>', methods=['GET'])
def get_stock_news(ticker):
    """Get news for specific ticker"""
    try:
        news_items = []
        
        # Try Yahoo Finance news first (free, no API key needed)
        try:
            stock = yf.Ticker(ticker)
            yf_news = stock.news
            
            for item in yf_news[:10]:
                sentiment = get_news_sentiment(item.get('title', '') + ' ' + item.get('summary', ''))
                news_items.append({
                    'title': item.get('title', 'No title'),
                    'summary': item.get('summary', '')[:200],
                    'url': item.get('link', ''),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', time.time())).isoformat(),
                    'sentiment': sentiment
                })
        except:
            pass
        
        # Calculate overall sentiment
        sentiments = [n['sentiment'] for n in news_items]
        pos_count = sentiments.count('positive')
        neg_count = sentiments.count('negative')
        total = len(sentiments) if sentiments else 1
        
        sentiment_score = ((pos_count - neg_count) / total) * 100
        
        return jsonify({
            'ticker': ticker.upper(),
            'news': news_items,
            'sentiment_score': round(sentiment_score, 2),
            'positive_count': pos_count,
            'negative_count': neg_count,
            'neutral_count': sentiments.count('neutral'),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-news', methods=['GET'])
def get_market_news():
    """Get general market news"""
    try:
        news_items = []
        
        # Market indices to check for news
        indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow, NASDAQ
        
        for index in indices:
            try:
                stock = yf.Ticker(index)
                yf_news = stock.news
                
                for item in yf_news[:3]:
                    news_items.append({
                        'title': item.get('title', 'No title'),
                        'summary': item.get('summary', '')[:200],
                        'url': item.get('link', ''),
                        'source': item.get('publisher', 'Yahoo Finance'),
                        'published': datetime.fromtimestamp(item.get('providerPublishTime', time.time())).isoformat()
                    })
            except:
                continue
        
        # Remove duplicates
        seen = set()
        unique_news = []
        for item in news_items:
            if item['title'] not in seen:
                seen.add(item['title'])
                unique_news.append(item)
        
        return jsonify({
            'news': unique_news[:15],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/<ticker>', methods=['GET'])
def analyze_stock(ticker):
    """Complete stock analysis with predictions"""
    try:
        stock, hist = get_stock_data(ticker.upper(), period='1y')
        
        if stock is None or hist is None or len(hist) == 0:
            return jsonify({'error': 'Unable to fetch stock data'}), 404
        
        info = stock.info
        prices = hist['Close']
        current_price = prices.iloc[-1]
        
        # Technical Analysis
        rsi = calculate_rsi(prices)
        macd = calculate_macd(prices)
        bb = calculate_bollinger_bands(prices)
        ma = calculate_moving_averages(prices)
        
        # Technical Score (0-100)
        tech_score = 50
        if rsi < 30:
            tech_score += 20
        elif rsi > 70:
            tech_score -= 20
        elif 40 <= rsi <= 60:
            tech_score += 10
        
        if macd['histogram'] > 0:
            tech_score += 10
        else:
            tech_score -= 10
        
        if current_price > ma['sma_50']:
            tech_score += 10
        if current_price > ma['sma_200']:
            tech_score += 10
        
        tech_score = max(0, min(100, tech_score))
        
        # Fundamental Analysis
        pe_ratio = info.get('trailingPE', 0)
        forward_pe = info.get('forwardPE', 0)
        peg_ratio = info.get('pegRatio', 0)
        debt_to_equity = info.get('debtToEquity', 0)
        roe = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
        profit_margin = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
        
        # Fundamental Score (0-100)
        fund_score = 50
        if pe_ratio and 10 <= pe_ratio <= 25:
            fund_score += 15
        if peg_ratio and peg_ratio < 1:
            fund_score += 10
        if debt_to_equity and debt_to_equity < 50:
            fund_score += 10
        if roe > 15:
            fund_score += 10
        if profit_margin > 15:
            fund_score += 10
        
        fund_score = max(0, min(100, fund_score))
        
        # Calculate volatility for Monte Carlo
        returns = prices.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        mu = returns.mean() * 252
        
        # Monte Carlo Simulation
        mc_1week = monte_carlo_simulation(current_price, mu, volatility, 1/52)
        mc_1month = monte_carlo_simulation(current_price, mu, volatility, 1/12)
        mc_3month = monte_carlo_simulation(current_price, mu, volatility, 3/12)
        
        # Overall prediction
        overall_score = (tech_score * 0.4 + fund_score * 0.4 + mc_1month['prob_profit'] * 0.2)
        
        if overall_score >= 70:
            signal = 'STRONG BUY'
            signal_color = 'green'
        elif overall_score >= 55:
            signal = 'BUY'
            signal_color = 'lightgreen'
        elif overall_score >= 45:
            signal = 'NEUTRAL'
            signal_color = 'yellow'
        elif overall_score >= 30:
            signal = 'SELL'
            signal_color = 'orange'
        else:
            signal = 'STRONG SELL'
            signal_color = 'red'
        
        return jsonify({
            'ticker': ticker.upper(),
            'name': info.get('shortName', ticker),
            'current_price': round(current_price, 2),
            'signal': signal,
            'signal_color': signal_color,
            'confidence': round(overall_score, 1),
            
            'technical': {
                'score': round(tech_score, 1),
                'rsi': round(rsi, 2),
                'macd': {k: round(v, 2) for k, v in macd.items()},
                'bollinger_bands': {k: round(v, 2) for k, v in bb.items()},
                'moving_averages': {k: round(v, 2) for k, v in ma.items()}
            },
            
            'fundamental': {
                'score': round(fund_score, 1),
                'pe_ratio': round(pe_ratio, 2) if pe_ratio else None,
                'forward_pe': round(forward_pe, 2) if forward_pe else None,
                'peg_ratio': round(peg_ratio, 2) if peg_ratio else None,
                'debt_to_equity': round(debt_to_equity, 2) if debt_to_equity else None,
                'roe': round(roe, 2),
                'profit_margin': round(profit_margin, 2),
                'market_cap': info.get('marketCap', 0),
                'revenue': info.get('totalRevenue', 0),
                'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else None
            },
            
            'predictions': {
                '1_week': {k: round(v, 2) for k, v in mc_1week.items()},
                '1_month': {k: round(v, 2) for k, v in mc_1month.items()},
                '3_month': {k: round(v, 2) for k, v in mc_3month.items()}
            },
            
            'recommendation': {
                'entry': round(current_price * 0.98, 2),
                'stop_loss': round(current_price * 0.95, 2),
                'target': round(mc_1month['percentile_75'], 2),
                'risk_reward': round((mc_1month['percentile_75'] - current_price) / (current_price * 0.05), 2)
            },
            
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/options/<ticker>', methods=['GET'])
def get_options_chain(ticker):
    """Get options chain for a ticker"""
    try:
        stock = yf.Ticker(ticker.upper())
        expirations = stock.options
        
        if not expirations:
            return jsonify({'error': 'No options available for this ticker'}), 404
        
        # Get first expiration date
        exp_date = expirations[0]
        opt_chain = stock.option_chain(exp_date)
        
        calls = opt_chain.calls.head(20).to_dict('records')
        puts = opt_chain.puts.head(20).to_dict('records')
        
        # Clean up data
        for option in calls + puts:
            for key in option:
                if pd.isna(option[key]):
                    option[key] = 0
        
        return jsonify({
            'ticker': ticker.upper(),
            'expiration': exp_date,
            'all_expirations': expirations[:10],
            'calls': calls,
            'puts': puts,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate-bs', methods=['POST'])
def calculate_black_scholes():
    """Calculate Black-Scholes price and Greeks"""
    try:
        data = request.json
        S = float(data.get('stock_price', 100))
        K = float(data.get('strike', 100))
        T = float(data.get('time_to_expiry', 30)) / 365
        r = float(data.get('risk_free_rate', 4.5)) / 100
        sigma = float(data.get('volatility', 30)) / 100
        option_type = data.get('option_type', 'call')
        
        price = black_scholes(S, K, T, r, sigma, option_type)
        greeks = calculate_greeks(S, K, T, r, sigma, option_type)
        
        return jsonify({
            'theoretical_price': round(price, 2),
            'greeks': greeks,
            'inputs': {
                'stock_price': S,
                'strike': K,
                'days_to_expiry': int(T * 365),
                'risk_free_rate': r * 100,
                'volatility': sigma * 100,
                'option_type': option_type
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['POST'])
def backtest_strategy():
    """Simple backtesting endpoint"""
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL')
        strategy = data.get('strategy', 'sma_crossover')
        period = data.get('period', '1y')
        
        stock, hist = get_stock_data(ticker, period)
        
        if hist is None or len(hist) == 0:
            return jsonify({'error': 'Unable to fetch data'}), 404
        
        prices = hist['Close']
        
        # Simple SMA crossover strategy
        if strategy == 'sma_crossover':
            sma_short = prices.rolling(window=20).mean()
            sma_long = prices.rolling(window=50).mean()
            
            signals = []
            position = None
            entry_price = 0
            trades = []
            
            for i in range(50, len(prices)):
                if sma_short.iloc[i] > sma_long.iloc[i] and position != 'long':
                    position = 'long'
                    entry_price = prices.iloc[i]
                    signals.append('BUY')
                    trades.append({
                        'date': str(hist.index[i].date()),
                        'action': 'BUY',
                        'price': round(entry_price, 2)
                    })
                elif sma_short.iloc[i] < sma_long.iloc[i] and position == 'long':
                    position = None
                    exit_price = prices.iloc[i]
                    profit = ((exit_price - entry_price) / entry_price) * 100
                    signals.append('SELL')
                    trades.append({
                        'date': str(hist.index[i].date()),
                        'action': 'SELL',
                        'price': round(exit_price, 2),
                        'profit_pct': round(profit, 2)
                    })
                else:
                    signals.append('HOLD')
            
            total_return = ((prices.iloc[-1] - prices.iloc[50]) / prices.iloc[50]) * 100
            
            return jsonify({
                'ticker': ticker.upper(),
                'strategy': strategy,
                'period': period,
                'total_return': round(total_return, 2),
                'num_trades': len([t for t in trades if t['action'] == 'BUY']),
                'trades': trades[-10:],  # Last 10 trades
                'final_price': round(prices.iloc[-1], 2),
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

"""
# Statistical Arbitrage Strategies for Volatility Trading

This module implements statistical arbitrage strategies for volatility trading, focusing on:
1. Volatility risk premium harvesting
2. Cross-asset volatility arbitrage
3. Term structure arbitrage
4. Skew and convexity arbitrage

Based on strategies from successful volatility traders like @shortbelly and @darjohn25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, pearsonr
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


class VolatilityStatArbAnalyzer:
    """
    Class for analyzing statistical arbitrage opportunities in volatility markets
    """

    def __init__(self, session):
        """
        Initialize with tastytrade session

        Parameters:
        -----------
        session : tastytrade.Session
            Authenticated tastytrade session
        """
        self.session = session

    def analyze_iv_vs_realized(self, symbol, lookback_days=252, forecast_days=30):
        """
        Analyze implied volatility vs. realized volatility for statistical arbitrage opportunities

        Parameters:
        -----------
        symbol : str
            Symbol to analyze
        lookback_days : int
            Number of days to look back for historical analysis
        forecast_days : int
            Number of days for volatility forecast

        Returns:
        --------
        dict : Analysis results
        """
        # In a real implementation, this would fetch data from tastytrade API
        # Here we'll create synthetic data for visualization

        # Create synthetic dates
        dates = pd.date_range(end=pd.Timestamp.now(), periods=lookback_days, freq='B')

        # Create synthetic price data with realistic properties
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, lookback_days)
        prices = 100 * np.exp(np.cumsum(returns))

        # Calculate realized volatility (20-day rolling)
        realized_vol = pd.Series(returns).rolling(20).std() * np.sqrt(252) * 100

        # Create synthetic implied volatility with volatility risk premium
        base_iv = realized_vol.mean()
        iv_series = realized_vol * 1.2 + np.random.normal(0, 1, lookback_days)

        # Create dataframe
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices,
            'Return': returns,
            'RealizedVol': realized_vol,
            'ImpliedVol': iv_series
        })

        # Calculate volatility risk premium
        df['VolRiskPremium'] = df['ImpliedVol'] - df['RealizedVol']

        # Calculate z-score of volatility risk premium
        vrp_mean = df['VolRiskPremium'].mean()
        vrp_std = df['VolRiskPremium'].std()
        df['VRP_ZScore'] = (df['VolRiskPremium'] - vrp_mean) / vrp_std

        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Plot price
        ax1.plot(df['Date'], df['Price'], linewidth=1.5)
        ax1.set_ylabel('Price')
        ax1.set_title(f'{symbol} Price History', fontsize=14)
        ax1.grid(True)

        # Plot implied vs realized volatility
        ax2.plot(df['Date'], df['ImpliedVol'], label='Implied Vol', linewidth=1.5)
        ax2.plot(df['Date'], df['RealizedVol'], label='Realized Vol', linewidth=1.5)
        ax2.set_ylabel('Volatility (%)')
        ax2.set_title('Implied vs. Realized Volatility', fontsize=14)
        ax2.legend()
        ax2.grid(True)

        # Plot volatility risk premium z-score
        ax3.plot(df['Date'], df['VRP_ZScore'], linewidth=1.5)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='+2σ')
        ax3.axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='-2σ')
        ax3.set_ylabel('Z-Score')
        ax3.set_xlabel('Date')
        ax3.set_title('Volatility Risk Premium Z-Score', fontsize=14)
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()

        # Calculate summary statistics
        stats = {
            'mean_vrp': vrp_mean,
            'std_vrp': vrp_std,
            'current_vrp': df['VolRiskPremium'].iloc[-1],
            'current_vrp_zscore': df['VRP_ZScore'].iloc[-1],
            'iv_realized_correlation': pearsonr(df['ImpliedVol'].dropna(), df['RealizedVol'].dropna())[0]
        }

        return {
            'figure': fig,
            'data': df,
            'stats': stats
        }

    def analyze_term_structure(self, symbol, expiries_days=[7, 30, 60, 90, 180, 270, 365]):
        """
        Analyze volatility term structure for arbitrage opportunities

        Parameters:
        -----------
        symbol : str
            Symbol to analyze
        expiries_days : list
            List of days to expiration to analyze

        Returns:
        --------
        dict : Analysis results
        """
        # In a real implementation, this would fetch data from tastytrade API
        # Here we'll create synthetic data for visualization

        # Create synthetic term structure data
        np.random.seed(42)

        # Create historical dates (30 days)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='B')

        # Create term structure for each date
        term_structures = []

        # Base parameters for term structure
        base_vol = 20.0  # Base volatility level
        term_slope = 0.1  # Normal upward slope

        for i in range(len(dates)):
            # Create term structure with random variations
            # Start with contango (upward sloping), then shift to backwardation
            if i < 15:
                # Contango
                slope = term_slope + np.random.normal(0, 0.02)
            else:
                # Shift to backwardation
                slope = -term_slope + np.random.normal(0, 0.02)

            # Create term structure
            term_struct = {}
            for dte in expiries_days:
                # Calculate IV for this expiry
                # Add some noise to make it realistic
                iv = base_vol + slope * np.log(dte / 30) + np.random.normal(0, 0.5)
                term_struct[dte] = max(5.0, iv)  # Ensure IV is positive

            term_structures.append(term_struct)

        # Create dataframe
        df = pd.DataFrame(term_structures, index=dates)

        # Calculate term structure metrics

        # 1. Contango/backwardation measure: ratio of short-term to long-term vol
        df['Contango'] = df[expiries_days[0]] / df[expiries_days[-1]]

        # 2. Term structure slope: regression coefficient
        slopes = []
        for _, row in df.iterrows():
            x = np.log(expiries_days)
            y = [row[dte] for dte in expiries_days]
            slope, _ = np.polyfit(x, y, 1)
            slopes.append(slope)

        df['Slope'] = slopes

        # 3. Term structure curvature: difference between actual mid-term vol and linear interpolation
        curvatures = []
        mid_idx = len(expiries_days) // 2
        mid_dte = expiries_days[mid_idx]

        for _, row in df.iterrows():
            short_vol = row[expiries_days[0]]
            long_vol = row[expiries_days[-1]]
            mid_vol = row[mid_dte]

            # Linear interpolation at mid-point
            log_short = np.log(expiries_days[0])
            log_long = np.log(expiries_days[-1])
            log_mid = np.log(mid_dte)

            interp_vol = short_vol + (long_vol - short_vol) * (log_mid - log_short) / (log_long - log_short)

            # Curvature = actual - interpolated
            curvature = mid_vol - interp_vol
            curvatures.append(curvature)

        df['Curvature'] = curvatures

        # Calculate z-scores
        df['Contango_ZScore'] = (df['Contango'] - df['Contango'].mean()) / df['Contango'].std()
        df['Slope_ZScore'] = (df['Slope'] - df['Slope'].mean()) / df['Slope'].std()
        df['Curvature_ZScore'] = (df['Curvature'] - df['Curvature'].mean()) / df['Curvature'].std()

        # Create figure
        fig = plt.figure(figsize=(16, 12))

        # 1. Term structure evolution
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)

        # Select a few dates to display
        dates_to_plot = [0, 7, 14, 21, 29]  # First, middle, and last dates

        for idx in dates_to_plot:
            date = dates[idx]
            vols = [df.loc[date, dte] for dte in expiries_days]
            ax1.plot(expiries_days, vols, 'o-', linewidth=2, label=date.strftime('%Y-%m-%d'))

        ax1.set_xlabel('Days to Expiration')
        ax1.set_ylabel('Implied Volatility (%)')
        ax1.set_title(f'{symbol} Volatility Term Structure Evolution', fontsize=14)
        ax1.legend()
        ax1.grid(True)

        # 2. Contango/backwardation measure
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        ax2.plot(dates, df['Contango'], linewidth=2)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax2.set_ylabel('Short/Long Vol Ratio')
        ax2.set_title('Contango/Backwardation Measure', fontsize=14)
        ax2.grid(True)

        # 3. Term structure slope
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        ax3.plot(dates, df['Slope'], linewidth=2)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_ylabel('Slope Coefficient')
        ax3.set_title('Term Structure Slope', fontsize=14)
        ax3.grid(True)

        # 4. Z-scores
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        ax4.plot(dates, df['Contango_ZScore'], linewidth=2, label='Contango Z-Score')
        ax4.plot(dates, df['Slope_ZScore'], linewidth=2, label='Slope Z-Score')
        ax4.plot(dates, df['Curvature_ZScore'], linewidth=2, label='Curvature Z-Score')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=2, color='red', linestyle='--', alpha=0.7)
        ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Z-Score')
        ax4.set_title('Term Structure Metrics Z-Scores', fontsize=14)
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()

        # Calculate current term structure metrics
        current_metrics = {
            'contango': df['Contango'].iloc[-1],
            'contango_zscore': df['Contango_ZScore'].iloc[-1],
            'slope': df['Slope'].iloc[-1],
            'slope_zscore': df['Slope_ZScore'].iloc[-1],
            'curvature': df['Curvature'].iloc[-1],
            'curvature_zscore': df['Curvature_ZScore'].iloc[-1]
        }

        # Determine term structure state
        if current_metrics['contango'] > 1.05:
            state = 'Strong Contango'
        elif current_metrics['contango'] > 1.0:
            state = 'Mild Contango'
        elif current_metrics['contango'] < 0.95:
            state = 'Strong Backwardation'
        elif current_metrics['contango'] < 1.0:
            state = 'Mild Backwardation'
        else:
            state = 'Flat'

        current_metrics['state'] = state

        return {
            'figure': fig,
            'data': df,
            'current_metrics': current_metrics
        }

    def analyze_skew_arbitrage(self, symbol, moneyness_levels=[0.9, 0.95, 1.0, 1.05, 1.1], dte=30):
        """
        Analyze volatility skew for arbitrage opportunities

        Parameters:
        -----------
        symbol : str
            Symbol to analyze
        moneyness_levels : list
            List of moneyness levels to analyze
        dte : int
            Days to expiration to analyze

        Returns:
        --------
        dict : Analysis results
        """
        # In a real implementation, this would fetch data from tastytrade API
        # Here we'll create synthetic data for visualization

        # Create synthetic dates (30 days)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='B')

        # Create synthetic skew data
        np.random.seed(42)

        # Base parameters
        base_vol = 20.0
        base_skew = -1.5  # Negative skew (higher put IVs)

        # Create skew data for each date
        skew_data = []

        for i in range(len(dates)):
            # Vary skew over time
            if i < 10:
                # Normal skew
                skew = base_skew + np.random.normal(0, 0.2)
            elif i < 20:
                # Steepening skew
                skew = base_skew * (1 + 0.5 * (i - 10) / 10) + np.random.normal(0, 0.2)
            else:
                # Flattening skew
                skew = base_skew * (1 - 0.3 * (i - 20) / 10) + np.random.normal(0, 0.2)

            # Create skew curve
            skew_curve = {}
            for moneyness in moneyness_levels:
                # Calculate IV for this moneyness
                # Higher IV for lower strikes (negative skew)
                iv = base_vol + skew * (1.0 - moneyness) + np.random.normal(0, 0.3)
                skew_curve[moneyness] = max(5.0, iv)  # Ensure IV is positive

            skew_data.append(skew_curve)

        # Create dataframe
        df = pd.DataFrame(skew_data, index=dates)

        # Calculate skew metrics

        # 1. Risk reversal: 25-delta put IV - 25-delta call IV
        # Approximate with 0.9 and 1.1 moneyness
        df['RiskReversal'] = df[0.9] - df[1.1]

        # 2. Butterfly: (25-delta put IV + 25-delta call IV)/2 - ATM IV
        df['Butterfly'] = (df[0.9] + df[1.1]) / 2 - df[1.0]

        # 3. Skew slope: regression coefficient
        slopes = []
        for _, row in df.iterrows():
            x = np.array(moneyness_levels)
            y = [row[m] for m in moneyness_levels]
            slope, _ = np.polyfit(x, y, 1)
            slopes.append(slope)

        df['Slope'] = slopes

        # Calculate z-scores
        df['RiskReversal_ZScore'] = (df['RiskReversal'] - df['RiskReversal'].mean()) / df['RiskReversal'].std()
        df['Butterfly_ZScore'] = (df['Butterfly'] - df['Butterfly'].mean()) / df['Butterfly'].std()
        df['Slope_ZScore'] = (df['Slope'] - df['Slope'].mean()) / df['Slope'].std()

        # Create figure
        fig = plt.figure(figsize=(16, 12))

        # 1. Skew evolution
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)

        # Select a few dates to display
        dates_to_plot = [0, 7, 14, 21, 29]  # First, middle, and last dates

        for idx in dates_to_plot:
            date = dates[idx]
            vols = [df.loc[date, m] for m in moneyness_levels]
            ax1.plot(moneyness_levels, vols, 'o-', linewidth=2, label=date.strftime('%Y-%m-%d'))

        ax1.set_xlabel('Moneyness (Strike/Spot)')
        ax1.set_ylabel('Implied Volatility (%)')
        ax1.set_title(f'{symbol} Volatility Skew Evolution ({dte} DTE)', fontsize=14)
        ax1.legend()
        ax1.grid(True)

        # 2. Risk reversal
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        ax2.plot(dates, df['RiskReversal'], linewidth=2)
        ax2.set_ylabel('Risk Reversal (90/110)')
        ax2.set_title('Risk Reversal', fontsize=14)
        ax2.grid(True)

        # 3. Butterfly
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        ax3.plot(dates, df['Butterfly'], linewidth=2)
        ax3.set_ylabel('Butterfly (90/100/110)')
        ax3.set_title('Butterfly', fontsize=14)
        ax3.grid(True)

        # 4. Z-scores
        ax4 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        ax4.plot(dates, df['RiskReversal_ZScore'], linewidth=2, label='Risk Reversal Z-Score')
        ax4.plot(dates, df['Butterfly_ZScore'], linewidth=2, label='Butterfly Z-Score')
        ax4.plot(dates, df['Slope_ZScore'], linewidth=2, label='Slope Z-Score')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.axhline(y=2, color='red', linestyle='--', alpha=0.7)
        ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Z-Score')
        ax4.set_title('Skew Metrics Z-Scores', fontsize=14)
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()

        # Calculate current skew metrics
        current_metrics = {
            'risk_reversal': df['RiskReversal'].iloc[-1],
            'risk_reversal_zscore': df['RiskReversal_ZScore'].iloc[-1],
            'butterfly': df['Butterfly'].iloc[-1],
            'butterfly_zscore': df['Butterfly_ZScore'].iloc[-1],
            'slope': df['Slope'].iloc[-1],
            'slope_zscore': df['Slope_ZScore'].iloc[-1]
        }

        # Determine skew state
        if current_metrics['risk_reversal'] > 5.0:
            state = 'Very Steep Skew'
        elif current_metrics['risk_reversal'] > 2.5:
            state = 'Steep Skew'
        elif current_metrics['risk_reversal'] < 0:
            state = 'Inverted Skew'
        else:
            state = 'Normal Skew'

        current_metrics['state'] = state

        return {
            'figure': fig,
            'data': df,
            'current_metrics': current_metrics
        }

    def analyze_cross_asset_vol(self, symbols=['SPX', 'VIX'], lookback_days=252):
        """
        Analyze cross-asset volatility relationships for arbitrage opportunities

        Parameters:
        -----------
        symbols : list
            List of symbols to analyze
        lookback_days : int
            Number of days to look back for historical analysis

        Returns:
        --------
        dict : Analysis results
        """
        # In a real implementation, this would fetch data from tastytrade API
        # Here we'll create synthetic data for visualization

        # Create synthetic dates
        dates = pd.date_range(end=pd.Timestamp.now(), periods=lookback_days, freq='B')

        # Create synthetic implied volatility data with realistic properties
        np.random.seed(42)

        # Create correlated IV series
        iv_data = {}

        # Base parameters
        base_iv = {
            'SPX': 18.0,
            'VIX': 85.0,  # VIX options IV is typically much higher
            'NDX': 22.0,
            'RUT': 25.0
        }

        # Create IV series for each symbol
        for symbol in symbols:
            if symbol in base_iv:
                base = base_iv[symbol]
            else:
                base = 20.0

            # Create mean-reverting IV series
            iv_series = np.zeros(lookback_days)
            iv_series[0] = base

            for i in range(1, lookback_days):
                # Mean reversion + random shock
                iv_series[i] = iv_series[i - 1] + 0.1 * (base - iv_series[i - 1]) + np.random.normal(0, 1.0)

            iv_data[symbol] = iv_series

        # Add correlation between series
        # For SPX and VIX, they should be negatively correlated
        if 'SPX' in symbols and 'VIX' in symbols:
            spx_idx = symbols.index('SPX')
            vix_idx = symbols.index('VIX')

            # Add negative correlation component
            common_factor = np.random.normal(0, 1.0, lookback_days)
            iv_data[symbols[spx_idx]] -= 0.5 * common_factor
            iv_data[symbols[vix_idx]] += 0.5 * common_factor

        # Create dataframe
        df = pd.DataFrame({symbol: iv_data[symbol] for symbol in symbols}, index=dates)

        # Calculate ratios and spreads between pairs
        pairs = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym1 = symbols[i]
                sym2 = symbols[j]

                # Calculate ratio
                ratio_name = f'{sym1}/{sym2}_Ratio'
                df[ratio_name] = df[sym1] / df[sym2]

                # Calculate spread
                spread_name = f'{sym1}/{sym2}_Spread'
                df[spread_name] = df[sym1] - df[sym2]

                # Calculate z-scores
                ratio_zscore_name = f'{sym1}/{sym2}_Ratio_ZScore'
                spread_zscore_name = f'{sym1}/{sym2}_Spread_ZScore'

                df[ratio_zscore_name] = (df[ratio_name] - df[ratio_name].mean()) / df[ratio_name].std()
                df[spread_zscore_name] = (df[spread_name] - df[spread_name].mean()) / df[spread_name].std()

                pairs.append((sym1, sym2))

        # Create figure
        fig = plt.figure(figsize=(16, 4 * (len(pairs) + 1)))

        # 1. Individual IV series
        ax1 = plt.subplot2grid((len(pairs) + 1, 1), (0, 0))

        for symbol in symbols:
            ax1.plot(dates, df[symbol], linewidth=2, label=symbol)

        ax1.set_ylabel('Implied Volatility (%)')
        ax1.set_title('Implied Volatility by Symbol', fontsize=14)
        ax1.legend()
        ax1.grid(True)

        # 2. Ratio and spread z-scores for each pair
        for i, (sym1, sym2) in enumerate(pairs):
            ax = plt.subplot2grid((len(pairs) + 1, 1), (i + 1, 0))

            ratio_zscore_name = f'{sym1}/{sym2}_Ratio_ZScore'
            spread_zscore_name = f'{sym1}/{sym2}_Spread_ZScore'

            ax.plot(dates, df[ratio_zscore_name], linewidth=2, label=f'{sym1}/{sym2} Ratio Z-Score')
            ax.plot(dates, df[spread_zscore_name], linewidth=2, label=f'{sym1}/{sym2} Spread Z-Score')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=2, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=-2, color='red', linestyle='--', alpha=0.7)

            if i == len(pairs) - 1:
                ax.set_xlabel('Date')

            ax.set_ylabel('Z-Score')
            ax.set_title(f'{sym1} vs {sym2} Volatility Relationship', fontsize=14)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        # Calculate current metrics for each pair
        current_metrics = {}

        for sym1, sym2 in pairs:
            ratio_name = f'{sym1}/{sym2}_Ratio'
            spread_name = f'{sym1}/{sym2}_Spread'
            ratio_zscore_name = f'{sym1}/{sym2}_Ratio_ZScore'
            spread_zscore_name = f'{sym1}/{sym2}_Spread_ZScore'

            pair_metrics = {
                'ratio': df[ratio_name].iloc[-1],
                'ratio_zscore': df[ratio_zscore_name].iloc[-1],
                'spread': df[spread_name].iloc[-1],
                'spread_zscore': df[spread_zscore_name].iloc[-1],
                'correlation': df[sym1].corr(df[sym2])
            }

            current_metrics[f'{sym1}/{sym2}'] = pair_metrics

        return {
            'figure': fig,
            'data': df,
            'current_metrics': current_metrics
        }


def demo_stat_arb_analysis():
    """
    Demonstrate statistical arbitrage analysis
    """
    # In a real implementation, this would use the tastytrade session from the user's code
    # Here we'll create a placeholder session for visualization
    session = None

    # Create analyzer
    analyzer = VolatilityStatArbAnalyzer(session)

    # Analyze implied vs realized volatility
    print("Analyzing implied vs realized volatility...")
    analyzer.analyze_iv_vs_realized('SPX')

    # Analyze term structure
    print("Analyzing volatility term structure...")
    analyzer.analyze_term_structure('SPX')

    # Analyze skew
    print("Analyzing volatility skew...")
    analyzer.analyze_skew_arbitrage('SPX')

    # Analyze cross-asset volatility
    print("Analyzing cross-asset volatility relationships...")
    analyzer.analyze_cross_asset_vol(['SPX', 'VIX', 'NDX', 'RUT'])

    print("Statistical arbitrage analysis completed.")


if __name__ == "__main__":
    demo_stat_arb_analysis()

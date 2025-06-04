"""
Volatility Surface and Greeks Evolution Visualization

This code provides visualization tools for:
1. Volatility surface across strikes and expirations
2. Greeks (delta, gamma, vega) evolution across the surface
3. Term structure analysis for volatility
4. Statistical arbitrage metrics between correlated assets

Based on strategies from successful volatility traders like @shortbelly and @darjohn25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.interpolate import griddata, interp2d
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12


class VolatilitySurfaceVisualizer:
    """
    Class for visualizing volatility surfaces and Greeks evolution
    """

    def __init__(self, session, symbol):
        """
        Initialize with tastytrade session and symbol

        Parameters:
        -----------
        session : tastytrade.Session
            Authenticated tastytrade session
        symbol : str
            Symbol to analyze (e.g., 'SPX', 'VIX')
        """
        self.session = session
        self.symbol = symbol
        self.option_chain = None
        self.underlying_price = None
        self.expiry_dates = None
        self.strikes = None
        self.iv_surface = None
        self.rate = 0.05  # Risk-free rate assumption

    def fetch_data(self):
        """
        Fetch option chain data from tastytrade
        """
        # In a real implementation, this would use the tastytrade SDK
        # Here we'll create a placeholder for the visualization code
        print(f"Fetching option chain data for {self.symbol}...")

        # For visualization purposes, we'll create synthetic data
        # In real implementation, replace with:
        # self.option_chain = get_option_chain(self.session, self.symbol)
        # self.underlying_price = get_market_data_by_type(self.session, indices=[self.symbol])['mark']

        self.underlying_price = 4200.0 if self.symbol == 'SPX' else 15.0

        # Create synthetic expiry dates (days to expiration)
        self.dte_values = np.array([7, 14, 30, 60, 90, 180, 270, 365])

        # Create synthetic strikes centered around current price
        price_range = 0.25 * self.underlying_price
        self.strikes = np.linspace(
            self.underlying_price - price_range,
            self.underlying_price + price_range,
            20
        )

        # Create moneyness grid (strike/spot)
        self.moneyness = np.array([strike / self.underlying_price for strike in self.strikes])

        # Create synthetic IV surface with smile and term structure
        self.iv_surface = np.zeros((len(self.dte_values), len(self.strikes)))

        for i, dte in enumerate(self.dte_values):
            # Base volatility increases with time to expiration (normal term structure)
            base_vol = 0.15 + 0.05 * np.log(dte / 30)

            # Create volatility smile
            for j, moneyness in enumerate(self.moneyness):
                # Skew - higher vol for lower strikes (risk aversion)
                skew_factor = 0.2 * (1.0 - moneyness) ** 2
                # Smile curvature
                smile_factor = 0.1 * (moneyness - 1.0) ** 2

                # Combine factors for final IV
                self.iv_surface[i, j] = max(0.05, base_vol + skew_factor + smile_factor)

        print("Data fetched and processed successfully.")
        return self

    def plot_volatility_surface(self):
        """
        Plot 3D volatility surface
        """
        if self.iv_surface is None:
            self.fetch_data()

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(self.moneyness, self.dte_values)

        # Plot the surface
        surf = ax.plot_surface(
            X, Y, self.iv_surface,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility')

        # Set labels and title
        ax.set_xlabel('Moneyness (Strike/Spot)')
        ax.set_ylabel('Days to Expiration')
        ax.set_zlabel('Implied Volatility')
        ax.set_title(f'{self.symbol} Implied Volatility Surface', fontsize=16)

        # Adjust view angle
        ax.view_init(30, 45)

        return fig

    def plot_volatility_term_structure(self):
        """
        Plot volatility term structure for different moneyness levels
        """
        if self.iv_surface is None:
            self.fetch_data()

        fig, ax = plt.subplots(figsize=(12, 8))

        # Select a few moneyness levels to display
        moneyness_levels = [0.9, 0.95, 1.0, 1.05, 1.1]
        moneyness_indices = [np.abs(self.moneyness - level).argmin() for level in moneyness_levels]

        # Plot term structure for each moneyness level
        for idx, level in zip(moneyness_indices, moneyness_levels):
            ax.plot(
                self.dte_values,
                self.iv_surface[:, idx],
                'o-',
                linewidth=2,
                label=f'Moneyness = {level:.2f}'
            )

        # Add labels and title
        ax.set_xlabel('Days to Expiration')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'{self.symbol} Volatility Term Structure', fontsize=16)
        ax.legend()
        ax.grid(True)

        return fig

    def plot_volatility_smile(self):
        """
        Plot volatility smile for different expirations
        """
        if self.iv_surface is None:
            self.fetch_data()

        fig, ax = plt.subplots(figsize=(12, 8))

        # Select a few expiries to display
        dte_indices = [0, 2, 4, 6]  # Corresponds to 7, 30, 90, 270 days
        dte_labels = [f'{self.dte_values[i]} DTE' for i in dte_indices]

        # Plot smile for each expiry
        for idx, label in zip(dte_indices, dte_labels):
            ax.plot(
                self.moneyness,
                self.iv_surface[idx, :],
                'o-',
                linewidth=2,
                label=label
            )

        # Add vertical line at ATM
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='ATM')

        # Add labels and title
        ax.set_xlabel('Moneyness (Strike/Spot)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'{self.symbol} Volatility Smile', fontsize=16)
        ax.legend()
        ax.grid(True)

        return fig

    def calculate_greeks(self, expiry_idx=2):
        """
        Calculate option Greeks for a specific expiration

        Parameters:
        -----------
        expiry_idx : int
            Index of expiration to use from self.dte_values

        Returns:
        --------
        dict : Dictionary containing delta, gamma, vega, theta arrays
        """
        if self.iv_surface is None:
            self.fetch_data()

        # Get time to expiration in years
        t = self.dte_values[expiry_idx] / 365.0

        # Initialize arrays for Greeks
        delta = np.zeros(len(self.strikes))
        gamma = np.zeros(len(self.strikes))
        vega = np.zeros(len(self.strikes))
        theta = np.zeros(len(self.strikes))

        # Calculate Greeks for each strike
        for i, strike in enumerate(self.strikes):
            # Extract volatility for this strike and expiry
            sigma = self.iv_surface[expiry_idx, i]

            # Calculate d1 and d2 from Black-Scholes
            d1 = (np.log(self.underlying_price / strike) +
                  (self.rate + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)

            # Calculate Greeks
            delta[i] = norm.cdf(d1)  # Call delta
            gamma[i] = norm.pdf(d1) / (self.underlying_price * sigma * np.sqrt(t))
            vega[i] = self.underlying_price * norm.pdf(d1) * np.sqrt(t) / 100  # Scaled for readability
            theta[i] = -(self.underlying_price * norm.pdf(d1) * sigma) / (2 * np.sqrt(t)) / 365  # Per day

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta
        }

    def plot_greeks_surface(self):
        """
        Plot 3D surfaces for delta, gamma, and vega across strikes and expirations
        """
        if self.iv_surface is None:
            self.fetch_data()

        # Initialize arrays for Greeks across all expirations
        delta_surface = np.zeros((len(self.dte_values), len(self.strikes)))
        gamma_surface = np.zeros((len(self.dte_values), len(self.strikes)))
        vega_surface = np.zeros((len(self.dte_values), len(self.strikes)))

        # Calculate Greeks for each expiration
        for i in range(len(self.dte_values)):
            greeks = self.calculate_greeks(expiry_idx=i)
            delta_surface[i, :] = greeks['delta']
            gamma_surface[i, :] = greeks['gamma']
            vega_surface[i, :] = greeks['vega']

        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(self.moneyness, self.dte_values)

        # Create figure with subplots
        fig = plt.figure(figsize=(18, 14))

        # Delta surface
        ax1 = fig.add_subplot(221, projection='3d')
        surf1 = ax1.plot_surface(
            X, Y, delta_surface,
            cmap=cm.viridis,
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )
        ax1.set_xlabel('Moneyness (Strike/Spot)')
        ax1.set_ylabel('Days to Expiration')
        ax1.set_zlabel('Delta')
        ax1.set_title('Delta Surface', fontsize=14)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

        # Gamma surface
        ax2 = fig.add_subplot(222, projection='3d')
        surf2 = ax2.plot_surface(
            X, Y, gamma_surface,
            cmap=cm.plasma,
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )
        ax2.set_xlabel('Moneyness (Strike/Spot)')
        ax2.set_ylabel('Days to Expiration')
        ax2.set_zlabel('Gamma')
        ax2.set_title('Gamma Surface', fontsize=14)
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

        # Vega surface
        ax3 = fig.add_subplot(223, projection='3d')
        surf3 = ax3.plot_surface(
            X, Y, vega_surface,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )
        ax3.set_xlabel('Moneyness (Strike/Spot)')
        ax3.set_ylabel('Days to Expiration')
        ax3.set_zlabel('Vega')
        ax3.set_title('Vega Surface', fontsize=14)
        fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=5)

        # Vega/Theta ratio (proxy for volatility risk premium)
        ax4 = fig.add_subplot(224)

        # Select a few expiries to display
        dte_indices = [0, 2, 4, 6]  # Corresponds to 7, 30, 90, 270 days
        dte_labels = [f'{self.dte_values[i]} DTE' for i in dte_indices]

        # Plot vega/gamma ratio for different expirations
        for idx, label in zip(dte_indices, dte_labels):
            greeks = self.calculate_greeks(expiry_idx=idx)
            # Calculate vega/gamma ratio (avoiding division by zero)
            vega_gamma_ratio = np.zeros_like(greeks['gamma'])
            mask = greeks['gamma'] > 1e-10
            vega_gamma_ratio[mask] = greeks['vega'][mask] / greeks['gamma'][mask]

            ax4.plot(
                self.moneyness,
                vega_gamma_ratio,
                'o-',
                linewidth=2,
                label=label
            )

        ax4.set_xlabel('Moneyness (Strike/Spot)')
        ax4.set_ylabel('Vega/Gamma Ratio')
        ax4.set_title('Vega/Gamma Ratio (Higher = More Convexity Exposure)', fontsize=14)
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.suptitle(f'{self.symbol} Greeks Evolution Across Volatility Surface', fontsize=18, y=1.02)

        return fig

    def plot_greeks_evolution(self, moneyness_level=1.0):
        """
        Plot how Greeks evolve over time for a specific moneyness level

        Parameters:
        -----------
        moneyness_level : float
            Moneyness level to analyze (default: 1.0 = ATM)
        """
        if self.iv_surface is None:
            self.fetch_data()

        # Find closest moneyness level
        moneyness_idx = np.abs(self.moneyness - moneyness_level).argmin()
        actual_moneyness = self.moneyness[moneyness_idx]

        # Initialize arrays for Greeks across all expirations
        delta_term = np.zeros(len(self.dte_values))
        gamma_term = np.zeros(len(self.dte_values))
        vega_term = np.zeros(len(self.dte_values))
        theta_term = np.zeros(len(self.dte_values))

        # Calculate Greeks for each expiration
        for i in range(len(self.dte_values)):
            greeks = self.calculate_greeks(expiry_idx=i)
            delta_term[i] = greeks['delta'][moneyness_idx]
            gamma_term[i] = greeks['gamma'][moneyness_idx]
            vega_term[i] = greeks['vega'][moneyness_idx]
            theta_term[i] = greeks['theta'][moneyness_idx]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Delta evolution
        axes[0, 0].plot(self.dte_values, delta_term, 'o-', linewidth=2, color='blue')
        axes[0, 0].set_xlabel('Days to Expiration')
        axes[0, 0].set_ylabel('Delta')
        axes[0, 0].set_title('Delta Term Structure')
        axes[0, 0].grid(True)

        # Gamma evolution
        axes[0, 1].plot(self.dte_values, gamma_term, 'o-', linewidth=2, color='green')
        axes[0, 1].set_xlabel('Days to Expiration')
        axes[0, 1].set_ylabel('Gamma')
        axes[0, 1].set_title('Gamma Term Structure')
        axes[0, 1].grid(True)

        # Vega evolution
        axes[1, 0].plot(self.dte_values, vega_term, 'o-', linewidth=2, color='red')
        axes[1, 0].set_xlabel('Days to Expiration')
        axes[1, 0].set_ylabel('Vega')
        axes[1, 0].set_title('Vega Term Structure')
        axes[1, 0].grid(True)

        # Theta evolution
        axes[1, 1].plot(self.dte_values, theta_term, 'o-', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Days to Expiration')
        axes[1, 1].set_ylabel('Theta (per day)')
        axes[1, 1].set_title('Theta Term Structure')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.suptitle(
            f'{self.symbol} Greeks Evolution Over Time (Moneyness = {actual_moneyness:.2f})',
            fontsize=16,
            y=1.02
        )

        return fig

    def plot_skew_metrics(self):
        """
        Plot risk reversal and butterfly metrics across expirations
        """
        if self.iv_surface is None:
            self.fetch_data()

        # Initialize arrays for skew metrics
        risk_reversal = np.zeros(len(self.dte_values))
        butterfly = np.zeros(len(self.dte_values))

        # Find indices for 90%, 100%, and 110% moneyness
        idx_90 = np.abs(self.moneyness - 0.9).argmin()
        idx_100 = np.abs(self.moneyness - 1.0).argmin()
        idx_110 = np.abs(self.moneyness - 1.1).argmin()

        # Calculate risk reversal and butterfly for each expiration
        for i in range(len(self.dte_values)):
            # Risk reversal = 25 delta call IV - 25 delta put IV
            # We'll approximate using 90% and 110% moneyness
            risk_reversal[i] = self.iv_surface[i, idx_110] - self.iv_surface[i, idx_90]

            # Butterfly = (25 delta call IV + 25 delta put IV)/2 - ATM IV
            # We'll approximate using 90%, 100%, and 110% moneyness
            butterfly[i] = (self.iv_surface[i, idx_90] + self.iv_surface[i, idx_110]) / 2 - self.iv_surface[i, idx_100]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Risk reversal plot
        ax1.plot(self.dte_values, risk_reversal, 'o-', linewidth=2, color='blue')
        ax1.set_xlabel('Days to Expiration')
        ax1.set_ylabel('Risk Reversal (90/110)')
        ax1.set_title('Risk Reversal Term Structure', fontsize=14)
        ax1.grid(True)

        # Butterfly plot
        ax2.plot(self.dte_values, butterfly, 'o-', linewidth=2, color='green')
        ax2.set_xlabel('Days to Expiration')
        ax2.set_ylabel('Butterfly (90/100/110)')
        ax2.set_title('Butterfly Term Structure', fontsize=14)
        ax2.grid(True)

        plt.tight_layout()
        plt.suptitle(f'{self.symbol} Volatility Skew Metrics', fontsize=16, y=1.02)

        return fig


class StatisticalArbitrageVisualizer:
    """
    Class for visualizing statistical arbitrage opportunities in volatility
    """

    def __init__(self, session, symbols):
        """
        Initialize with tastytrade session and symbols

        Parameters:
        -----------
        session : tastytrade.Session
            Authenticated tastytrade session
        symbols : list
            List of symbols to analyze (e.g., ['SPX', 'VIX'])
        """
        self.session = session
        self.symbols = symbols
        self.vol_surfaces = {}

    def fetch_data(self):
        """
        Fetch volatility surface data for all symbols
        """
        for symbol in self.symbols:
            print(f"Fetching data for {symbol}...")
            vol_viz = VolatilitySurfaceVisualizer(self.session, symbol)
            vol_viz.fetch_data()
            self.vol_surfaces[symbol] = vol_viz

        return self

    def plot_iv_correlation(self, dte_idx=2):
        """
        Plot correlation between implied volatilities of different symbols

        Parameters:
        -----------
        dte_idx : int
            Index of days to expiration to use
        """
        if not self.vol_surfaces:
            self.fetch_data()

        if len(self.symbols) < 2:
            raise ValueError("Need at least two symbols for correlation analysis")

        fig, ax = plt.subplots(figsize=(10, 8))

        # Get common moneyness range
        moneyness_values = self.vol_surfaces[self.symbols[0]].moneyness

        # Plot IV for each symbol
        for symbol in self.symbols:
            iv_slice = self.vol_surfaces[symbol].iv_surface[dte_idx, :]
            ax.plot(
                moneyness_values,
                iv_slice,
                'o-',
                linewidth=2,
                label=symbol
            )

        # Add vertical line at ATM
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='ATM')

        # Add labels and title
        ax.set_xlabel('Moneyness (Strike/Spot)')
        ax.set_ylabel('Implied Volatility')
        dte = self.vol_surfaces[self.symbols[0]].dte_values[dte_idx]
        ax.set_title(f'Implied Volatility Comparison ({dte} DTE)', fontsize=16)
        ax.legend()
        ax.grid(True)

        return fig

    def plot_iv_ratio(self, symbol1, symbol2):
        """
        Plot ratio of implied volatilities between two symbols

        Parameters:
        -----------
        symbol1 : str
            First symbol
        symbol2 : str
            Second symbol
        """
        if not self.vol_surfaces:
            self.fetch_data()

        if symbol1 not in self.symbols or symbol2 not in self.symbols:
            raise ValueError(f"Symbols must be in {self.symbols}")

        # Get volatility surfaces
        vol1 = self.vol_surfaces[symbol1]
        vol2 = self.vol_surfaces[symbol2]

        # Calculate IV ratio for each expiration and strike
        iv_ratio = np.zeros((len(vol1.dte_values), len(vol1.moneyness)))

        for i in range(len(vol1.dte_values)):
            iv_ratio[i, :] = vol1.iv_surface[i, :] / vol2.iv_surface[i, :]

        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(vol1.moneyness, vol1.dte_values)

        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(
            X, Y, iv_ratio,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=f'{symbol1}/{symbol2} IV Ratio')

        # Set labels and title
        ax.set_xlabel('Moneyness (Strike/Spot)')
        ax.set_ylabel('Days to Expiration')
        ax.set_zlabel(f'{symbol1}/{symbol2} IV Ratio')
        ax.set_title(f'Implied Volatility Ratio Surface: {symbol1}/{symbol2}', fontsize=16)

        # Adjust view angle
        ax.view_init(30, 45)

        return fig

    def plot_iv_spread(self, symbol1, symbol2, dte_indices=[0, 2, 4, 6]):
        """
        Plot spread of implied volatilities between two symbols for different expirations

        Parameters:
        -----------
        symbol1 : str
            First symbol
        symbol2 : str
            Second symbol
        dte_indices : list
            Indices of days to expiration to plot
        """
        if not self.vol_surfaces:
            self.fetch_data()

        if symbol1 not in self.symbols or symbol2 not in self.symbols:
            raise ValueError(f"Symbols must be in {self.symbols}")

        # Get volatility surfaces
        vol1 = self.vol_surfaces[symbol1]
        vol2 = self.vol_surfaces[symbol2]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get DTE labels
        dte_labels = [f'{vol1.dte_values[i]} DTE' for i in dte_indices]

        # Plot IV spread for each expiration
        for idx, label in zip(dte_indices, dte_labels):
            iv_spread = vol1.iv_surface[idx, :] - vol2.iv_surface[idx, :]

            ax.plot(
                vol1.moneyness,
                iv_spread,
                'o-',
                linewidth=2,
                label=label
            )

        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Add vertical line at ATM
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)

        # Add labels and title
        ax.set_xlabel('Moneyness (Strike/Spot)')
        ax.set_ylabel('IV Spread')
        ax.set_title(f'Implied Volatility Spread: {symbol1} - {symbol2}', fontsize=16)
        ax.legend()
        ax.grid(True)

        return fig

    def plot_historical_iv_ratio(self):
        """
        Plot historical IV ratio between symbols (synthetic data for visualization)
        """
        # This would normally use historical data from tastytrade API
        # Here we'll create synthetic data for visualization purposes

        # Create synthetic dates
        dates = pd.date_range(end=pd.Timestamp.now(), periods=252, freq='B')

        # Create synthetic IV ratio time series with mean reversion
        np.random.seed(42)
        ratio_mean = 1.5
        ratio_std = 0.2
        mean_reversion_speed = 0.05

        ratio = np.zeros(len(dates))
        ratio[0] = ratio_mean

        for i in range(1, len(dates)):
            shock = np.random.normal(0, ratio_std)
            ratio[i] = ratio[i - 1] + mean_reversion_speed * (ratio_mean - ratio[i - 1]) + shock

        # Create dataframe
        df = pd.DataFrame({
            'Date': dates,
            'IV_Ratio': ratio
        })

        # Calculate z-score
        rolling_mean = df['IV_Ratio'].rolling(window=20).mean()
        rolling_std = df['IV_Ratio'].rolling(window=20).std()
        df['Z_Score'] = (df['IV_Ratio'] - rolling_mean) / rolling_std

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot IV ratio
        ax1.plot(df['Date'], df['IV_Ratio'], linewidth=2)
        ax1.axhline(y=ratio_mean, color='red', linestyle='--', alpha=0.7, label='Mean')
        ax1.axhline(y=ratio_mean + 2 * ratio_std, color='green', linestyle='--', alpha=0.7, label='+2 Std')
        ax1.axhline(y=ratio_mean - 2 * ratio_std, color='green', linestyle='--', alpha=0.7, label='-2 Std')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('IV Ratio')
        ax1.set_title(f'Historical IV Ratio: {self.symbols[0]}/{self.symbols[1]}', fontsize=14)
        ax1.legend()
        ax1.grid(True)

        # Plot z-score
        ax2.plot(df['Date'], df['Z_Score'], linewidth=2)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=2, color='green', linestyle='--', alpha=0.7, label='+2 Z')
        ax2.axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='-2 Z')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Z-Score (20-day)')
        ax2.set_title('Z-Score of IV Ratio (Statistical Arbitrage Signal)', fontsize=14)
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        return fig


class SABRModelVisualizer:
    """
    Class for fitting and visualizing SABR model on options surface
    """

    def __init__(self, session, symbol):
        """
        Initialize with tastytrade session and symbol

        Parameters:
        -----------
        session : tastytrade.Session
            Authenticated tastytrade session
        symbol : str
            Symbol to analyze (e.g., 'SPX', 'VIX')
        """
        self.session = session
        self.symbol = symbol
        self.vol_surface = None
        self.sabr_params = {}

    def fetch_data(self):
        """
        Fetch volatility surface data
        """
        vol_viz = VolatilitySurfaceVisualizer(self.session, self.symbol)
        vol_viz.fetch_data()
        self.vol_surface = vol_viz

        return self

    def fit_sabr_model(self):
        """
        Fit SABR model to volatility surface
        """
        if self.vol_surface is None:
            self.fetch_data()

        # In a real implementation, this would use optimization to fit SABR parameters
        # Here we'll create synthetic parameters for visualization

        # Initialize SABR parameters for each expiration
        for i, dte in enumerate(self.vol_surface.dte_values):
            # alpha: base volatility level
            alpha = 0.15 + 0.03 * np.log(dte / 30)

            # beta: CEV parameter (0 = normal, 1 = lognormal)
            beta = 0.7

            # rho: correlation between price and vol
            rho = -0.5 - 0.1 * np.log(dte / 30)

            # nu: volatility of volatility
            nu = 0.4 + 0.05 * np.log(dte / 30)

            self.sabr_params[dte] = {
                'alpha': alpha,
                'beta': beta,
                'rho': rho,
                'nu': nu
            }

        return self

    def sabr_implied_vol(self, f, k, t, alpha, beta, rho, nu):
        """
        Calculate SABR implied volatility

        Parameters:
        -----------
        f : float
            Forward price
        k : float
            Strike price
        t : float
            Time to expiration in years
        alpha, beta, rho, nu : float
            SABR parameters

        Returns:
        --------
        float : Implied volatility
        """
        # Handle ATM case separately
        if abs(f - k) < 1e-10:
            # ATM SABR formula
            return alpha * (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / (f ** (2 - 2 * beta)) +
                                 0.25 * rho * beta * nu * alpha / (f ** (1 - beta)) +
                                 ((2 - 3 * rho ** 2) / 24) * nu ** 2) * t)

        # Handle non-ATM case
        z = (nu / alpha) * ((f * k) ** (0.5 * (1 - beta))) * np.log(f / k)
        chi = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        # Calculate implied volatility
        return (alpha * ((f * k) ** ((beta - 1) / 2)) *
                (1 + ((1 - beta) ** 2 / 24) * (np.log(f / k)) ** 2 +
                 ((1 - beta) ** 4 / 1920) * (np.log(f / k)) ** 4) *
                z / chi *
                (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / ((f * k) ** (1 - beta)) +
                      0.25 * rho * beta * nu * alpha / ((f * k) ** ((1 - beta) / 2)) +
                      ((2 - 3 * rho ** 2) / 24) * nu ** 2) * t))

    def plot_sabr_fit(self, dte_indices=[0, 2, 4, 6]):
        """
        Plot SABR model fit against market implied volatilities

        Parameters:
        -----------
        dte_indices : list
            Indices of days to expiration to plot
        """
        if not self.sabr_params:
            self.fit_sabr_model()

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Get forward price (underlying price for simplicity)
        f = self.vol_surface.underlying_price

        # Plot for each selected expiration
        for i, idx in enumerate(dte_indices):
            if i >= len(axes):
                break

            dte = self.vol_surface.dte_values[idx]
            t = dte / 365.0  # Convert to years

            # Get SABR parameters for this expiration
            params = self.sabr_params[dte]

            # Get market implied volatilities
            market_iv = self.vol_surface.iv_surface[idx, :]

            # Calculate SABR implied volatilities
            strikes = self.vol_surface.strikes
            sabr_iv = np.zeros_like(strikes)

            for j, k in enumerate(strikes):
                sabr_iv[j] = self.sabr_implied_vol(
                    f, k, t,
                    params['alpha'], params['beta'], params['rho'], params['nu']
                )

            # Calculate residuals
            residuals = market_iv - sabr_iv

            # Plot market vs SABR
            ax = axes[i]
            ax.plot(self.vol_surface.moneyness, market_iv, 'o', label='Market')
            ax.plot(self.vol_surface.moneyness, sabr_iv, '-', linewidth=2, label='SABR')

            # Add residuals as bar plot
            ax_twin = ax.twinx()
            ax_twin.bar(self.vol_surface.moneyness, residuals, alpha=0.3, color='red', label='Residuals')
            ax_twin.set_ylabel('Residuals', color='red')
            ax_twin.tick_params(axis='y', labelcolor='red')
            ax_twin.axhline(y=0, color='red', linestyle='--', alpha=0.5)

            # Add labels and title
            ax.set_xlabel('Moneyness (Strike/Spot)')
            ax.set_ylabel('Implied Volatility')
            ax.set_title(
                f'{dte} DTE - SABR Fit (α={params["alpha"]:.3f}, β={params["beta"]:.2f}, ρ={params["rho"]:.2f}, ν={params["nu"]:.2f})')
            ax.legend(loc='upper left')
            ax.grid(True)

        plt.tight_layout()
        plt.suptitle(f'{self.symbol} SABR Model Fit to Volatility Surface', fontsize=16, y=1.02)

        return fig

    def plot_sabr_parameters(self):
        """
        Plot SABR parameters across expirations
        """
        if not self.sabr_params:
            self.fit_sabr_model()

        # Extract parameters for each expiration
        dtes = np.array(list(self.sabr_params.keys()))
        alphas = np.array([self.sabr_params[dte]['alpha'] for dte in dtes])
        betas = np.array([self.sabr_params[dte]['beta'] for dte in dtes])
        rhos = np.array([self.sabr_params[dte]['rho'] for dte in dtes])
        nus = np.array([self.sabr_params[dte]['nu'] for dte in dtes])

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Alpha plot
        axes[0, 0].plot(dtes, alphas, 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Days to Expiration')
        axes[0, 0].set_ylabel('Alpha (α)')
        axes[0, 0].set_title('SABR Alpha Parameter (Base Volatility)')
        axes[0, 0].grid(True)

        # Beta plot
        axes[0, 1].plot(dtes, betas, 'o-', linewidth=2)
        axes[0, 1].set_xlabel('Days to Expiration')
        axes[0, 1].set_ylabel('Beta (β)')
        axes[0, 1].set_title('SABR Beta Parameter (CEV Parameter)')
        axes[0, 1].grid(True)

        # Rho plot
        axes[1, 0].plot(dtes, rhos, 'o-', linewidth=2)
        axes[1, 0].set_xlabel('Days to Expiration')
        axes[1, 0].set_ylabel('Rho (ρ)')
        axes[1, 0].set_title('SABR Rho Parameter (Price-Vol Correlation)')
        axes[1, 0].grid(True)

        # Nu plot
        axes[1, 1].plot(dtes, nus, 'o-', linewidth=2)
        axes[1, 1].set_xlabel('Days to Expiration')
        axes[1, 1].set_ylabel('Nu (ν)')
        axes[1, 1].set_title('SABR Nu Parameter (Vol of Vol)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.suptitle(f'{self.symbol} SABR Parameters Across Term Structure', fontsize=16, y=1.02)

        return fig


def main():
    """
    Main function to demonstrate visualization capabilities
    """
    # In a real implementation, this would use the tastytrade session from the user's code
    # Here we'll create a placeholder session for visualization
    session = None

    # Create volatility surface visualizer for SPX
    print("Creating SPX volatility surface visualizer...")
    spx_viz = VolatilitySurfaceVisualizer(session, 'SPX')

    # Generate and display visualizations
    print("Generating volatility surface plot...")
    spx_viz.plot_volatility_surface()

    print("Generating volatility smile plot...")
    spx_viz.plot_volatility_smile()

    print("Generating volatility term structure plot...")
    spx_viz.plot_volatility_term_structure()

    print("Generating Greeks surface plot...")
    spx_viz.plot_greeks_surface()

    print("Generating Greeks evolution plot...")
    spx_viz.plot_greeks_evolution()

    print("Generating skew metrics plot...")
    spx_viz.plot_skew_metrics()

    # Create statistical arbitrage visualizer for SPX and VIX
    print("Creating statistical arbitrage visualizer...")
    stat_arb_viz = StatisticalArbitrageVisualizer(session, ['SPX', 'VIX'])

    print("Generating IV correlation plot...")
    stat_arb_viz.plot_iv_correlation()

    print("Generating IV ratio plot...")
    stat_arb_viz.plot_iv_ratio('SPX', 'VIX')

    print("Generating IV spread plot...")
    stat_arb_viz.plot_iv_spread('SPX', 'VIX')

    print("Generating historical IV ratio plot...")
    stat_arb_viz.plot_historical_iv_ratio()

    # Create SABR model visualizer
    print("Creating SABR model visualizer...")
    sabr_viz = SABRModelVisualizer(session, 'SPX')

    print("Generating SABR fit plot...")
    sabr_viz.plot_sabr_fit()

    print("Generating SABR parameters plot...")
    sabr_viz.plot_sabr_parameters()

    print("All visualizations generated successfully.")


if __name__ == "__main__":
    main()

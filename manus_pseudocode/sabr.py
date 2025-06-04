# SABR Model Implementation for Volatility Surface Analysis

"""
This module implements the SABR (Stochastic Alpha Beta Rho) model for volatility surface analysis.
It provides tools for:
1. Fitting the SABR model to market implied volatilities
2. Analyzing the fitted parameters across strikes and expirations
3. Visualizing the model fit and parameter evolution

Based on strategies from successful volatility traders like @shortbelly and @darjohn25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class SABRModel:
    """
    Implementation of the SABR stochastic volatility model
    """

    def __init__(self, forward, time_to_expiry):
        """
        Initialize SABR model

        Parameters:
        -----------
        forward : float
            Forward price
        time_to_expiry : float
            Time to expiry in years
        """
        self.forward = forward
        self.time_to_expiry = time_to_expiry
        self.params = {
            'alpha': 0.2,  # Initial volatility
            'beta': 0.7,  # CEV parameter (0 = normal, 1 = lognormal)
            'rho': -0.3,  # Correlation between price and vol
            'nu': 0.4  # Volatility of volatility
        }

    def implied_volatility(self, strike):
        """
        Calculate SABR implied volatility for a given strike

        Parameters:
        -----------
        strike : float or array-like
            Option strike price(s)

        Returns:
        --------
        float or array-like : Implied volatility
        """
        # Extract parameters
        alpha = self.params['alpha']
        beta = self.params['beta']
        rho = self.params['rho']
        nu = self.params['nu']
        f = self.forward
        t = self.time_to_expiry

        # Handle array input
        if hasattr(strike, '__iter__'):
            return np.array([self.implied_volatility(k) for k in strike])

        # Handle ATM case separately
        if abs(f - strike) < 1e-10:
            # ATM SABR formula
            return alpha * (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / (f ** (2 - 2 * beta)) +
                                 0.25 * rho * beta * nu * alpha / (f ** (1 - beta)) +
                                 ((2 - 3 * rho ** 2) / 24) * nu ** 2) * t)

        # Handle non-ATM case
        z = (nu / alpha) * ((f * strike) ** (0.5 * (1 - beta))) * np.log(f / strike)

        # Handle extreme cases
        if abs(z) < 1e-10:
            chi = 1.0
        else:
            chi = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        # Calculate implied volatility
        iv = (alpha * ((f * strike) ** ((beta - 1) / 2)) *
              (1 + ((1 - beta) ** 2 / 24) * (np.log(f / strike)) ** 2 +
               ((1 - beta) ** 4 / 1920) * (np.log(f / strike)) ** 4))

        if abs(z) < 1e-10:
            iv_final = iv
        else:
            iv_final = iv * (z / chi)

        # Apply time-dependent correction
        iv_final *= (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / ((f * strike) ** (1 - beta)) +
                          0.25 * rho * beta * nu * alpha / ((f * strike) ** ((1 - beta) / 2)) +
                          ((2 - 3 * rho ** 2) / 24) * nu ** 2) * t)

        return iv_final

    def fit(self, strikes, market_ivs, bounds=None):
        """
        Fit SABR parameters to market implied volatilities

        Parameters:
        -----------
        strikes : array-like
            Option strike prices
        market_ivs : array-like
            Market implied volatilities
        bounds : tuple of tuples, optional
            Bounds for parameters (alpha, beta, rho, nu)

        Returns:
        --------
        dict : Fitted parameters
        """
        # Default bounds
        if bounds is None:
            bounds = (
                (0.001, 1.0),  # alpha: (0, 1)
                (0.01, 0.99),  # beta: (0, 1)
                (-0.999, 0.999),  # rho: (-1, 1)
                (0.001, 2.0)  # nu: (0, 2)
            )

        # Initial parameters
        x0 = [self.params['alpha'], self.params['beta'], self.params['rho'], self.params['nu']]

        # Objective function: sum of squared errors
        def objective(x):
            self.params['alpha'] = x[0]
            self.params['beta'] = x[1]
            self.params['rho'] = x[2]
            self.params['nu'] = x[3]

            model_ivs = self.implied_volatility(strikes)
            return np.sum((model_ivs - market_ivs) ** 2)

        # Perform optimization
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

        # Update parameters with fitted values
        self.params['alpha'] = result.x[0]
        self.params['beta'] = result.x[1]
        self.params['rho'] = result.x[2]
        self.params['nu'] = result.x[3]

        return self.params

    def calibrate_surface(self, strikes_surface, market_ivs_surface, expiries):
        """
        Calibrate SABR parameters across the entire volatility surface

        Parameters:
        -----------
        strikes_surface : list of arrays
            Strike prices for each expiry
        market_ivs_surface : list of arrays
            Market implied volatilities for each expiry
        expiries : array-like
            Expiry times in years

        Returns:
        --------
        dict : Dictionary of fitted parameters for each expiry
        """
        params_by_expiry = {}

        for i, t in enumerate(expiries):
            self.time_to_expiry = t
            strikes = strikes_surface[i]
            market_ivs = market_ivs_surface[i]

            self.fit(strikes, market_ivs)
            params_by_expiry[t] = self.params.copy()

        return params_by_expiry


class SABRVisualizer:
    """
    Class for visualizing SABR model fits and parameters
    """

    def __init__(self, forward, symbol=''):
        """
        Initialize visualizer

        Parameters:
        -----------
        forward : float
            Forward price
        symbol : str, optional
            Symbol for plot titles
        """
        self.forward = forward
        self.symbol = symbol

    def plot_sabr_fit(self, strikes, market_ivs, sabr_params, time_to_expiry, title_suffix=''):
        """
        Plot SABR model fit against market implied volatilities

        Parameters:
        -----------
        strikes : array-like
            Option strike prices
        market_ivs : array-like
            Market implied volatilities
        sabr_params : dict
            SABR parameters (alpha, beta, rho, nu)
        time_to_expiry : float
            Time to expiry in years
        title_suffix : str, optional
            Additional text for plot title

        Returns:
        --------
        matplotlib.figure.Figure : Plot figure
        """
        # Create SABR model with fitted parameters
        model = SABRModel(self.forward, time_to_expiry)
        model.params = sabr_params.copy()

        # Calculate model implied volatilities
        model_ivs = model.implied_volatility(strikes)

        # Calculate residuals
        residuals = market_ivs - model_ivs

        # Create figure
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Calculate moneyness
        moneyness = strikes / self.forward

        # Plot market vs SABR
        ax1.plot(moneyness, market_ivs, 'o', label='Market')
        ax1.plot(moneyness, model_ivs, '-', linewidth=2, label='SABR')

        # Add residuals as bar plot
        ax2 = ax1.twinx()
        ax2.bar(moneyness, residuals, alpha=0.3, color='red', label='Residuals')
        ax2.set_ylabel('Residuals', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # Add labels and title
        ax1.set_xlabel('Moneyness (Strike/Spot)')
        ax1.set_ylabel('Implied Volatility')

        # Format time to expiry for display
        if time_to_expiry < 1 / 12:
            dte_str = f"{int(time_to_expiry * 365)} DTE"
        elif time_to_expiry < 1:
            dte_str = f"{int(time_to_expiry * 12)} months"
        else:
            dte_str = f"{time_to_expiry:.1f} years"

        title = f'{self.symbol} SABR Fit - {dte_str}'
        if title_suffix:
            title += f' - {title_suffix}'

        ax1.set_title(title)

        # Add parameter values to plot
        param_text = (f"α = {sabr_params['alpha']:.4f}, β = {sabr_params['beta']:.4f}, "
                      f"ρ = {sabr_params['rho']:.4f}, ν = {sabr_params['nu']:.4f}")
        ax1.text(0.05, 0.05, param_text, transform=ax1.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

        ax1.legend(loc='upper left')
        ax1.grid(True)

        plt.tight_layout()

        return fig

    def plot_parameter_evolution(self, expiries, params_by_expiry):
        """
        Plot evolution of SABR parameters across expiries

        Parameters:
        -----------
        expiries : array-like
            Expiry times in years
        params_by_expiry : dict
            Dictionary of fitted parameters for each expiry

        Returns:
        --------
        matplotlib.figure.Figure : Plot figure
        """
        # Extract parameters for each expiration
        alphas = np.array([params_by_expiry[t]['alpha'] for t in expiries])
        betas = np.array([params_by_expiry[t]['beta'] for t in expiries])
        rhos = np.array([params_by_expiry[t]['rho'] for t in expiries])
        nus = np.array([params_by_expiry[t]['nu'] for t in expiries])

        # Convert expiries to days for better visualization
        expiries_days = np.array(expiries) * 365

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Alpha plot
        axes[0, 0].plot(expiries_days, alphas, 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Days to Expiration')
        axes[0, 0].set_ylabel('Alpha (α)')
        axes[0, 0].set_title('SABR Alpha Parameter (Base Volatility)')
        axes[0, 0].grid(True)

        # Beta plot
        axes[0, 1].plot(expiries_days, betas, 'o-', linewidth=2)
        axes[0, 1].set_xlabel('Days to Expiration')
        axes[0, 1].set_ylabel('Beta (β)')
        axes[0, 1].set_title('SABR Beta Parameter (CEV Parameter)')
        axes[0, 1].grid(True)

        # Rho plot
        axes[1, 0].plot(expiries_days, rhos, 'o-', linewidth=2)
        axes[1, 0].set_xlabel('Days to Expiration')
        axes[1, 0].set_ylabel('Rho (ρ)')
        axes[1, 0].set_title('SABR Rho Parameter (Price-Vol Correlation)')
        axes[1, 0].grid(True)

        # Nu plot
        axes[1, 1].plot(expiries_days, nus, 'o-', linewidth=2)
        axes[1, 1].set_xlabel('Days to Expiration')
        axes[1, 1].set_ylabel('Nu (ν)')
        axes[1, 1].set_title('SABR Nu Parameter (Vol of Vol)')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.suptitle(f'{self.symbol} SABR Parameters Across Term Structure', fontsize=16, y=1.02)

        return fig

    def plot_forward_volatility(self, expiries, params_by_expiry, moneyness_levels=[0.9, 0.95, 1.0, 1.05, 1.1]):
        """
        Plot forward volatility implied by SABR parameters

        Parameters:
        -----------
        expiries : array-like
            Expiry times in years
        params_by_expiry : dict
            Dictionary of fitted parameters for each expiry
        moneyness_levels : list, optional
            Moneyness levels to plot

        Returns:
        --------
        matplotlib.figure.Figure : Plot figure
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Convert expiries to days for better visualization
        expiries_days = np.array(expiries) * 365

        # Calculate forward volatilities for each moneyness level
        for moneyness in moneyness_levels:
            strike = moneyness * self.forward
            forward_vols = []

            for t in expiries:
                model = SABRModel(self.forward, t)
                model.params = params_by_expiry[t]
                iv = model.implied_volatility(strike)
                forward_vols.append(iv)

            ax.plot(expiries_days, forward_vols, 'o-', linewidth=2, label=f'K/F = {moneyness:.2f}')

        # Add labels and title
        ax.set_xlabel('Days to Expiration')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'{self.symbol} Forward Volatility Term Structure')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()

        return fig

    def plot_local_volatility(self, expiries, params_by_expiry, moneyness_range=(0.7, 1.3), num_points=50):
        """
        Plot local volatility surface implied by SABR parameters

        Parameters:
        -----------
        expiries : array-like
            Expiry times in years
        params_by_expiry : dict
            Dictionary of fitted parameters for each expiry
        moneyness_range : tuple, optional
            Range of moneyness values to plot
        num_points : int, optional
            Number of moneyness points to calculate

        Returns:
        --------
        matplotlib.figure.Figure : Plot figure
        """
        # Create moneyness grid
        moneyness = np.linspace(moneyness_range[0], moneyness_range[1], num_points)
        strikes = moneyness * self.forward

        # Create time grid (convert to days for better visualization)
        expiries_days = np.array(expiries) * 365

        # Calculate local volatility surface
        local_vol_surface = np.zeros((len(expiries), len(strikes)))

        for i, t in enumerate(expiries):
            for j, k in enumerate(strikes):
                # Calculate local volatility using SABR parameters
                # This is an approximation of local volatility
                model = SABRModel(self.forward, t)
                model.params = params_by_expiry[t]

                # Get implied volatility
                iv = model.implied_volatility(k)

                # Extract parameters
                alpha = model.params['alpha']
                beta = model.params['beta']
                rho = model.params['rho']
                nu = model.params['nu']

                # Calculate local volatility (approximation)
                if abs(k - self.forward) < 1e-10:
                    # ATM local vol
                    local_vol = alpha * (1 + (((1 - beta) ** 2 / 24) * alpha ** 2 / (self.forward ** (2 - 2 * beta)) +
                                              ((2 - 3 * rho ** 2) / 24) * nu ** 2) * t)
                else:
                    # Non-ATM local vol (approximation)
                    local_vol = iv * (1 - 0.5 * rho * nu * (k - self.forward) / (iv * k))

                local_vol_surface[i, j] = local_vol

        # Create meshgrid for 3D surface
        X, Y = np.meshgrid(moneyness, expiries_days)

        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(
            X, Y, local_vol_surface,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=True,
            alpha=0.8
        )

        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Local Volatility')

        # Set labels and title
        ax.set_xlabel('Moneyness (Strike/Spot)')
        ax.set_ylabel('Days to Expiration')
        ax.set_zlabel('Local Volatility')
        ax.set_title(f'{self.symbol} Local Volatility Surface', fontsize=16)

        # Adjust view angle
        ax.view_init(30, 45)

        return fig


def demo_sabr_model():
    """
    Demonstrate SABR model fitting and visualization
    """
    # Create synthetic data
    forward = 4200.0  # SPX price
    expiries = np.array([7 / 365, 30 / 365, 90 / 365, 180 / 365, 365 / 365])  # in years

    # Create strikes centered around forward price
    strikes_pct = np.linspace(0.8, 1.2, 20)
    strikes = forward * strikes_pct

    # Create synthetic market implied volatilities with smile
    market_ivs_surface = []

    for t in expiries:
        base_vol = 0.15 + 0.05 * np.log(t * 365 / 30)  # Term structure

        market_ivs = []
        for k in strikes:
            moneyness = k / forward
            # Skew - higher vol for lower strikes (risk aversion)
            skew_factor = 0.2 * (1.0 - moneyness) ** 2
            # Smile curvature
            smile_factor = 0.1 * (moneyness - 1.0) ** 2

            # Combine factors for final IV
            iv = max(0.05, base_vol + skew_factor + smile_factor)
            market_ivs.append(iv)

        market_ivs_surface.append(np.array(market_ivs))

    # Create SABR model and fit to market data
    sabr_model = SABRModel(forward, expiries[0])

    # Calibrate across the entire surface
    strikes_surface = [strikes] * len(expiries)
    params_by_expiry = sabr_model.calibrate_surface(strikes_surface, market_ivs_surface, expiries)

    # Create visualizer
    visualizer = SABRVisualizer(forward, 'SPX')

    # Plot SABR fit for each expiry
    for i, t in enumerate(expiries):
        dte = int(t * 365)
        visualizer.plot_sabr_fit(
            strikes,
            market_ivs_surface[i],
            params_by_expiry[t],
            t,
            f'{dte} DTE'
        )

    # Plot parameter evolution
    visualizer.plot_parameter_evolution(expiries, params_by_expiry)

    # Plot forward volatility
    visualizer.plot_forward_volatility(expiries, params_by_expiry)

    # Plot local volatility surface
    visualizer.plot_local_volatility(expiries, params_by_expiry)

    print("SABR model demonstration completed.")


if __name__ == "__main__":
    demo_sabr_model()

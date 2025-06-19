import pandas as pd
import numpy as np
from tastytrade import Account, Session, DXLinkStreamer
from tastytrade.dxfeed import Quote, Candle
from tastytrade.instruments import *
from tastytrade.market_data import *
from itertools import chain
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.ndimage import gaussian_filter1d


class OptionMethods:

    @staticmethod
    def batch_findoption(session, items, func, batch_size=90):
        # generator of batches
        batches = (
            items[i: i + batch_size]
            for i in range(0, len(items), batch_size)
        )
        return list(chain.from_iterable(func(session, [option.symbol for option in batch]) for batch in batches))

    @staticmethod
    def fetch_options(session, option_batch):
        optiondata = get_market_data_by_type(session, options=option_batch)
        return optiondata

    @staticmethod
    def convertchain(session, chain):
        all_results = OptionMethods.batch_findoption(session, chain, OptionMethods.fetch_options, batch_size=90)

        res_map = {res.symbol: res for res in all_results}

        combined = [
            {
                "symbol": opt.symbol,
                "strike_price": float(opt.strike_price),
                "option_type": opt.option_type,
                "bid": float(getattr(res_map.get(opt.symbol, None), "bid", np.nan)),
                "ask": float(getattr(res_map.get(opt.symbol, None), "ask", np.nan)),
            }
            for opt in chain
        ]

        df = pd.DataFrame(combined)
        vc = df["strike_price"].value_counts()
        # find any strikes that don’t have exactly 2 entries
        wrong = vc[vc != 2]

        # assert, printing out the offending strike_price values
        assert wrong.empty, f"Strikes without exactly two entries: {wrong.index.tolist()}"

        df["type"] = df["option_type"].apply(
            lambda ot: "C" if ot == OptionType.CALL
            else "P" if ot == OptionType.PUT
            else (_ for _ in ()).throw(ValueError(f"Unexpected OptionType: {ot!r}"))
        )

        df_pivot = (
            df
            .pivot(index="strike_price", columns="type", values=["bid", "ask"])
            .reset_index()
        )

        df_pivot.columns = [
            f"{col_type.lower()}{val}" if isinstance(col_type, str)
            else col_type  # this picks up the strike_price index as-is
            for val, col_type in df_pivot.columns
        ]

        # 7. Rename strike_price → strike, reorder
        dfchain = (
            df_pivot
            .rename(columns={"strike_price": "strike"})
            [["cbid", "cask", "strike", "pbid", "pask"]]
        )
        return dfchain

    @staticmethod
    def find_atmf_strike(chain):
        c = chain.copy()
        c['cmid'] = (c['cbid'] + c['cask']) / 2
        c['pmid'] = (c['pbid'] + c['pask']) / 2
        atmfs = c.loc[abs(c['cmid'] - c['pmid']).idxmin(), "strike"]
        return atmfs

    #TODO: all below
    @staticmethod
    def find_implied_dist(chain, dte, rfr, mode="butterfly", gaussian_sigma=2):
        if mode == "butterfly":
            return
        elif mode == "breedenlitzenberger":
            return
        else:
            return None

    @staticmethod
    def find_ivs(chain, dte, rfr, div_yield=0, gaussian_sigma=2):
        '''
        chain: pd DF with columns: cbid, cask, pbid, pask, strike
        dte in days, currently only supports "dirty IV" ie /365
        rfr in percentage points, ie do not normalize before passing
        div yield in percentage points
        gaussian_sigma is the sigma for the gaussian filter applied to the IVs
        '''
        if 'iv' in chain.columns:
            print('IVs already present in chain')
            return chain

        T = dte/365
        rfr /= 100
        div_yield /= 100

        df = chain.copy()
        if 'cmid' not in df.columns or 'pmid' not in df.columns:
            df['cmid'] = (df['cbid'] + df['cask']) / 2
            df['pmid'] = (df['pbid'] + df['pask']) / 2

        K_atm = OptionMethods.find_atmf_strike(df)
        atm_row = df.loc[df['strike'] == K_atm].iloc[0]
        cmid_atm, pmid_atm = atm_row['cmid'], atm_row['pmid']
        F = K_atm + np.exp((rfr-div_yield) * T) * (cmid_atm - pmid_atm)
        print(F)

        civ_list, piv_list = [], []
        for _, row in df.iterrows():
            K = row['strike']
            civ = OptionMethods.bs_iv(row['cmid'], F, K, T, rfr, is_call=True)
            piv = OptionMethods.bs_iv(row['pmid'], F, K, T, rfr, is_call=False)
            civ_list.append(civ)
            piv_list.append(piv)

        df['civ'] = civ_list
        df['piv'] = piv_list
        if gaussian_sigma > 0:
            df['civ'] = gaussian_filter1d(df['civ'], sigma=gaussian_sigma, mode='nearest')
            df['piv'] = gaussian_filter1d(df['piv'], sigma=gaussian_sigma, mode='nearest')
        return df

    @staticmethod
    def bs_call(F, K, T, sigma, rfr):
        if sigma <= 0 or T <= 0:
            return max(F - K, 0.0) * np.exp(-rfr * T)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-rfr * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))

    @staticmethod
    def bs_put(F, K, T, sigma, rfr):
        if sigma <= 0 or T <= 0:
            return max(K - F, 0.0) * np.exp(-rfr * T)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return np.exp(-rfr * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    @staticmethod
    def bs_iv(price, F, K, T, rfr, is_call=True):
        if price < 1e-8 or T <= 0:
            return 0.0

        def objective(sigma):
            if is_call:
                return OptionMethods.bs_call(F, K, T, sigma, rfr) - price
            else:
                return OptionMethods.bs_put(F, K, T, sigma, rfr) - price

        try:
            iv = brentq(objective, 1e-6, 5.0, maxiter=500)
        except ValueError:
            iv = 0.0
        return iv * 100


class DatabaseRequests:
    def __init__(self, project_url, api_key, table_name="market_snapshots"):
        self.project_url = project_url
        self.api_key = api_key
        self.table_name = table_name

    def skew_snapshot(self, date, period):
        return

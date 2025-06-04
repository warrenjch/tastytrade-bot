import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from auxiliary import *


class EFG:

    @staticmethod
    def calc_forward_iv(session, chain, dte, rfr, spot):  # need specifically the chain of the day
        T = dte / 365
        r = rfr / 100
        forward = spot * np.exp(r * T)
        chain = [opt for opt in chain if opt.settlement_type == "PM"]
        chain = OptionMethods.convertchain(session, chain)
        chain = chain.dropna()

        K0 = chain.loc[abs(chain["strike"] - forward).idxmin(), "strike"]

        otmcalls = chain[chain["strike"] >= K0]
        otmputs = chain[chain["strike"] <= K0]
        call_bound = otmcalls[
            ((otmcalls["cbid"] == 0) & (otmcalls.shift(-1)["cbid"] == 0))
            | ((otmcalls["cask"] == 0) & (otmcalls.shift(-1)["cask"] == 0))
            | (otmcalls["cbid"].isna())
            | (otmcalls["cask"].isna())
            ].index.min()
        put_bound = otmputs[
            ((otmputs["pbid"] == 0) & (otmputs.shift(1)["pbid"] == 0))
            | ((otmputs["pask"] == 0) & (otmputs.shift(1)["pask"] == 0))
            | (otmputs["pbid"].isna())
            | (otmputs["pask"].isna())
            ].index.max()

        if pd.notna(call_bound):
            chain = chain.loc[: call_bound - 1]  # right end is inclusive in pd loc
        if pd.notna(put_bound):
            chain = chain.loc[put_bound + 1:]
        if otmcalls.empty or otmputs.empty:
            raise ValueError(
                f"no OTM options check code for {dte} dte"
            )

        chain = chain[
            ~(
                    (
                            (chain["strike"] >= K0)
                            & ((chain["cbid"] == 0) | (chain["cask"] == 0))
                    )
                    | (
                            (chain["strike"] <= K0)
                            & ((chain["pbid"] == 0) | (chain["pask"] == 0))
                    )
            )
        ]  # ignoring rows that have either 0 bid or ask

        chain["dK"] = (chain["strike"].shift(-1) - chain["strike"].shift(1)) / 2
        chain.loc[chain.index[0], "dK"] = (
                chain.loc[chain.index[1], "strike"] - chain.loc[chain.index[0], "strike"]
        )
        chain.loc[chain.index[-1], "dK"] = (
                chain.loc[chain.index[-1], "strike"]
                - chain.loc[chain.index[-2], "strike"]
        )
        otmcalls = chain[chain["strike"] >= K0]
        otmputs = chain[chain["strike"] <= K0]

        call_vals = (
                (
                        otmcalls[otmcalls["strike"] != K0]["cbid"]
                        + otmcalls[otmcalls["strike"] != K0]["cask"]
                )
                / 2
                * (
                        otmcalls[otmcalls["strike"] != K0]["dK"]
                        / otmcalls[otmcalls["strike"] != K0]["strike"] ** 2
                )
        )
        put_vals = (
                (
                        otmputs[otmputs["strike"] != K0]["pbid"]
                        + otmputs[otmputs["strike"] != K0]["pask"]
                )
                / 2
                * (
                        otmputs[otmputs["strike"] != K0]["dK"]
                        / otmputs[otmputs["strike"] != K0]["strike"] ** 2
                )
        )
        forwardiv = (
                2
                / T
                * np.exp(r * T)
                * (
                        np.sum(call_vals)
                        + EFG.interpolate(otmcalls, "call", "fiv")
                        + np.sum(put_vals)
                        + EFG.interpolate(otmputs, "put", "fiv")
                        + (otmcalls.iloc[0]["cbid"] + otmcalls.iloc[0]["cask"])
                        / 4
                        * (otmcalls.iloc[0]["dK"] / otmcalls.iloc[0]["strike"] ** 2)
                        + (otmputs.iloc[-1]["pbid"] + otmputs.iloc[-1]["pask"])
                        / 4
                        * (otmputs.iloc[-1]["dK"] / otmputs.iloc[-1]["strike"] ** 2)
                )
                - 1 / T * (forward / K0 - 1) ** 2
        )

        return (
                10000 * forwardiv
        )  # 100^2, since varq is measured on vix options, therefore this also needs to be scaled to actual vix

    @staticmethod
    def calc_varq(session, chain, dte, rfr, forward):
        # in varq the chain is the vix chain and spot is spot vix
        calls_bounded, puts_bounded = False, False
        T = dte / 365
        r = rfr / 100
        chain = OptionMethods.convertchain(session, chain)
        chain = chain.dropna()
        chain["dK"] = (chain["strike"].shift(-1) - chain["strike"].shift(1)) / 2
        chain.loc[chain.index[0], "dK"] = (
                chain.loc[chain.index[1], "strike"] - chain.loc[chain.index[0], "strike"]
        )
        chain.loc[chain.index[-1], "dK"] = (
                chain.loc[chain.index[-1], "strike"]
                - chain.loc[chain.index[-2], "strike"]
        )
        K0 = chain.loc[abs(chain["strike"] - forward).idxmin(), "strike"]
        otmcalls = chain[chain["strike"] >= K0]
        otmputs = chain[chain["strike"] <= K0]
        call_bound = otmcalls[
            (otmcalls["cbid"] == 0)
            | (otmcalls["cask"] == 0)
            | (otmcalls["cbid"].isna())
            | (otmcalls["cask"].isna())
            ].index.min()
        put_bound = otmputs[
            (otmputs["pbid"] == 0)
            | (otmputs["pask"] == 0)
            | (otmputs["pbid"].isna())
            | (otmputs["pask"].isna())
            ].index.max()
        if pd.notna(call_bound):
            otmcalls = otmcalls.loc[: call_bound - 1]
            calls_bounded = True
        if pd.notna(put_bound):
            otmputs = otmputs.loc[put_bound + 1:]
            puts_bounded = True
        if calls_bounded == False:
            print("vix calls unbounded")
        if puts_bounded == False:
            print("vix puts unbounded")

        if otmcalls.empty and otmputs.empty:
            raise ValueError(
                f"no OTM options on VIX check code for {dte} dte"
            )
        elif otmputs.empty and not otmcalls.empty:
            # print(f"no OTM puts on VIX on {chain.loc[:,' [QUOTE_DATE]'].iloc[0]} exp {chain.loc[:,' [EXPIRE_DATE]'].iloc[0]}")
            call_vals = (otmcalls["cbid"] + otmcalls["cask"]) / 2 * otmcalls["dK"]
            varq = 2 * np.exp(r * T) * (
                        np.sum(call_vals) + (EFG.interpolate(otmcalls, "call", "varq") if not calls_bounded else 0))
        elif otmcalls.empty and not otmputs.empty:
            put_vals = (otmputs["pbid"] + otmputs["pask"]) / 2 * otmputs["dK"]
            varq = 2 * np.exp(r * T) * (
                        np.sum(put_vals) + (EFG.interpolate(otmputs, "put", "varq") if not puts_bounded else 0))
        else:
            call_vals = (
                    (
                            otmcalls[otmcalls["strike"] != K0]["cbid"]
                            + otmcalls[otmcalls["strike"] != K0]["cask"]
                    )
                    / 2
                    * otmcalls[otmcalls["strike"] != K0]["dK"]
            )
            put_vals = (
                    (
                            otmputs[otmputs["strike"] != K0]["pbid"]
                            + otmputs[otmputs["strike"] != K0]["pask"]
                    )
                    / 2
                    * otmputs[otmputs["strike"] != K0]["dK"]
            )
            varq = (
                    2
                    * np.exp(r * T)
                    * (
                            np.sum(call_vals)
                            + np.sum(put_vals)
                            + (EFG.interpolate(otmcalls, "call", "varq") if not calls_bounded else 0)
                            + (EFG.interpolate(otmputs, "put", "varq") if not puts_bounded else 0)
                            + (otmcalls.iloc[0]["cbid"] + otmcalls.iloc[0]["cask"])
                            / 4
                            * otmcalls.iloc[0]["dK"]
                            + (otmputs.iloc[-1]["pbid"] + otmputs.iloc[-1]["pask"])
                            / 4
                            * otmputs.iloc[-1]["dK"]
                    )
            )
        return varq

    @staticmethod
    def interpolate(chain, type, component):
        # sometimes, the chain gets "cut off" because there are not enough strikes. this results in an incorrect calculation of the implied vix future value
        # there are no easy workarounds. here are the few things we can do:
        # 1. ignore expiry if we cannot find call_bound or put_bound. however sometimes we cannot simply skip an expiry day (sometimes we do not have the luxury of finding another expiry day that works for our vx expiry)
        # 2. choosing only monthlies, but this is considerably less accurate, and in older years sometimes even monthlies do not have reliable strikes
        # 3. interpolating otm option prices, but this introduces errors due to assumptions being used in the distribution of prices. however, this is the most reasonable approach as of now
        # this interpolation function will attempt to estimate the size of the tail that is "cut off" from the option chain, and return 0 if the tail is not cut off (ie bound exists)

        # print(f'interpolating {chain.loc[:," [QUOTE_DATE]"].iloc[0]} exp {chain.loc[:," [EXPIRE_DATE]"].iloc[0]}')
        if len(chain) == 0:
            raise ValueError(
                f"interpolating null chain"
            )
        elif len(chain) == 1:
            print(
                f"unable to interpolate chain with 1 data point"
            )
            return 0
        else:
            if type == "call":
                cutoff = chain.iloc[-1]
                cutoffminus1 = chain.iloc[-2]
                c1 = cutoff["strike"]
                f1 = (cutoff["cbid"] + cutoff["cask"]) / 2
                x = cutoff["strike"] - cutoffminus1["strike"]
                y = (cutoff["cbid"] + cutoff["cask"]) / 2 - (
                        cutoffminus1["cbid"] + cutoffminus1["cask"]
                ) / 2
                if round(y, 3) >= 0:
                    f1 = (cutoffminus1["cbid"] + cutoffminus1["cask"]) / 2
                    y = -f1
                if component == "fiv":
                    tail = max(-(f1 ** 2 * x) / (2 * c1 ** 2 * y), 0)
                elif component == "varq":
                    tail = max(f1 - c1 * f1 - (f1 ** 2 * x) / (2 * y), 0)
                else:
                    raise ValueError(f"invalid component {component}")
            elif type == "put":
                cutoff = chain.iloc[0]
                cutoffplus1 = chain.iloc[1]
                c1 = cutoff["strike"]
                f1 = (cutoff["pbid"] + cutoff["pask"]) / 2
                x = cutoff["strike"] - cutoffplus1["strike"]
                y = (cutoff["pbid"] + cutoff["pask"]) / 2 - (
                        cutoffplus1["pbid"] + cutoffplus1["pask"]
                ) / 2
                if round(y, 3) >= 0:
                    f1 = (cutoffplus1["pbid"] + cutoffplus1["pask"]) / 2
                    y = f1
                if component == "fiv":
                    tail = max((f1 ** 2 * x) / (2 * c1 ** 2 * y), 0)
                elif component == "varq":
                    tail = max(-(f1 - c1 * f1 - (f1 ** 2 * x) / (2 * y)), 0)
                else:
                    raise ValueError(f"invalid component {component}")
            else:
                raise ValueError(f"invalid type {type}")
            # print(f'{dte} dte {type} adj tail {tail * 2/dte * np.exp(rfr * dte) * 10000} bound {cutoff[' [STRIKE]']} terms {(y/x) - (y*c1/x + y/2)/(c1 + x/2) - (y/(2*x)) * ((x/2 + f1 * (x/y))/(c1 + x/2))**2}, {2/dte}, {rfr} ')
            return tail
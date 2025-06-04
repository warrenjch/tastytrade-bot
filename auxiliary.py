import pandas as pd
import numpy as np
from tastytrade import Account, Session, DXLinkStreamer
from tastytrade.dxfeed import Quote, Candle
from tastytrade.instruments import *
from tastytrade.market_data import *
from itertools import chain


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
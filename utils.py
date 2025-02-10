from datetime import datetime, date
import time
import pandas_market_calendars as mcal


def unix_to_time(t: int, granularity: str = 'd', scaled: bool = True):
    if scaled:
        dt = datetime.fromtimestamp(t/1000)
    else:
        dt = datetime.fromtimestamp(t)

    if granularity == 'd':
        t = date(dt.year, dt.month, dt.day)
        return t
    elif granularity == 's':
        return dt
    else:
        raise ValueError(f'invalid granularity: {granularity}')
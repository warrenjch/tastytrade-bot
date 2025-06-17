import pandas as pd
import numpy as np
from tastytrade import Account, Session, DXLinkStreamer
from tastytrade.dxfeed import Quote, Candle
from tastytrade.instruments import *
from tastytrade.market_data import *
from tastytrade.market_sessions import ExchangeType, get_market_sessions, MarketStatus
import yfinance as yf
from curl_cffi import requests as yfrequests
from itertools import chain
from config import *
from alphas import *
from auxiliary import *
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import asyncio
import requests
import logging


session=None
account=None
yfsession = yfrequests.Session(impersonate="chrome")
nytime = datetime.now(ZoneInfo("America/New_York"))

async def setup():
    global session, account, config
    # setup
    config = Config(test=False)
    session = Session(config.username, config.password, is_test=config.test)
    account = await Account.a_get(session, config.account_number)

    # account info
    balance = account.get_balances(session)
    positions = account.get_positions(session)
    history = account.get_history(session)

asyncio.run(setup())
print("Session and account setup complete.")

streamer = DXLinkStreamer(session) #async streamer

# Configuration variables - replace with your actual values
TELEGRAM_BOT_TOKEN = config.tg_token  # Get this from @BotFather on Telegram
TELEGRAM_CHAT_ID = config.tg_chat_id  # Your chat ID or channel name (e.g., "@your_channel")


def send_to_telegram(message):
    """
    Send message to Telegram bot
    Args:
        message (str): The message to send
    """
    try:
        # Telegram has a 4096 character limit per message
        max_length = 4000  # Leave some buffer

        if len(message) <= max_length:
            _send_single_message(message)
        else:
            # Split into multiple messages if too long
            parts = [message[i:i + max_length] for i in range(0, len(message), max_length)]
            for i, part in enumerate(parts):
                header = f"Message part {i + 1}/{len(parts)}:\n\n" if len(parts) > 1 else ""
                _send_single_message(header + part)

        print("✅ Message sent to Telegram successfully")

    except Exception as e:
        print(f"❌ Failed to send message to Telegram: {str(e)}")
        logging.error(f"Telegram error: {str(e)}")
    return


def _send_single_message(text):
    """Send a single message to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
        'parse_mode': 'HTML'  # Allows basic HTML formatting
    }

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()  # Raises an exception for bad status codes
    return

def efg():
    SPXoptionchain = get_option_chain(session, "SPX")
    VIXoptionchain = get_option_chain(session, "VIX")

    SPXexpiries = list(SPXoptionchain.keys())
    VIXexpiries = list(VIXoptionchain.keys())
    #TODO: VXfutures ordering by expiry date
    #TODO: some way to push this data at 4am to tg
    VXfutures = Future.get(session, symbols = None, product_codes=["VX"])
    VXfutures = sorted(VXfutures, key=lambda x: x.last_trade_date)
    VXfm = VXfutures[0]
    VXbm = VXfutures[1]
    VXfmdata = get_market_data_by_type(session,futures=[VXfm.symbol])
    VXbmdata = get_market_data_by_type(session,futures=[VXbm.symbol])

    indexreq = ["SPX"]
    SPXdata = get_market_data_by_type(session,indices = indexreq)
    print("Data fetched.")

    vx1expiryf = VXfm.last_trade_date
    vx1expiryb = vx1expiryf + timedelta(days=30)
    vx2expiryf = VXbm.last_trade_date
    vx2expiryb = vx2expiryf + timedelta(days=30)
    try:
        treasury_bill = yf.Ticker("^IRX", session=yfsession)
        hist = treasury_bill.history(period="1d")
        rfr = hist['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching risk-free rate: {e}")
        print("Using default rate of 4.5%")
        rfr = 4.5

    today = date.today()
    ervart1values = []
    ervart2values = []
    values = {
            "ervart1": pd.NA,
            "varqt1": pd.NA,
            "eqv1": pd.NA,
            "ervart2": pd.NA,
            "varqt2": pd.NA,
            "eqv2": pd.NA,
        }

    spot = float(SPXdata[0].last)
    VXstyle = "last"
    if VXstyle == "last":
        vx1close = float(VXfmdata[0].last)
        vx2close = float(VXbmdata[0].last)
    elif VXstyle == "close":
        vx1close = float(VXfmdata[0].close)
        vx2close = float(VXbmdata[0].close)
    elif VXstyle == "prev-close":
        vx1close = float(VXfmdata[0].prev_close)
        vx2close = float(VXbmdata[0].prev_close)
    else:
        raise ValueError(f"invalid style {VXstyle}")

    print('calc efg1')
    for vx1expiry in [vx1expiryf, vx1expiryb]:
        t1 = (vx1expiry - today).days
        if vx1expiry in SPXexpiries:
            print('calc ervar:', vx1expiry)
            t1chain = SPXoptionchain[vx1expiry]

            ervart1 = EFG.calc_forward_iv(
                session, t1chain, t1, rfr, spot
            )
        else:
            if (
                SPXexpiries[0] > vx1expiry
            ):  # sometimes the closest vx futures expiry is before the closest spx option expiry, this happens in earlier years
                exp1 = today  # the workaround is to let the expiry be today to "simulate" a 0dte that expired at close today
            else:
                exp1 = max([d for d in SPXexpiries if d < vx1expiry], default=None)
            exp2 = min([d for d in SPXexpiries if d > vx1expiry], default=None)
            print('calc ervar:', exp1, exp2)
            t1e1 = (exp1 - today).days  # front expiry of this vix future
            t1e2 = (
                exp2 - today
            ).days  # back expiry of this vix future, does NOT refer to the expiry of either the front or back timestamps in vx1expiryf or b
            if date == exp1:
                vart1e1 = 0
                # print("date = exp1 for vx1") #debug line
            else:
                t1e1chain = SPXoptionchain[exp1]
                vart1e1 = EFG.calc_forward_iv(
                    session, t1e1chain, t1e1, rfr, spot
                )
            t1e2chain = SPXoptionchain[exp2]
            vart1e2 = EFG.calc_forward_iv(
                session, t1e2chain, t1e2, rfr, spot
            )
            ervart1 = ((t1e2 - t1) * t1e1 * vart1e1 + (t1 - t1e1) * t1e2 * vart1e2) / (
                (t1e2 - t1e1) * t1
            )  # https://gregorygundersen.com/blog/2023/09/10/deriving-vix/
        ervart1values.append((t1, ervart1))
    ervart1 = (
        ervart1values[1][0] * ervart1values[1][1]
        - ervart1values[0][0] * ervart1values[0][1]
    ) / (
        ervart1values[1][0] - ervart1values[0][0]
    )  # (Tb * vb - Ta * va) / (Tb - Ta)
    values["ervart1"] = ervart1

    assert vx1expiryf in VIXexpiries
    t1vixchain = VIXoptionchain[vx1expiryf]
    print('calc varq:', vx1expiryf)
    varqt1 = EFG.calc_varq(session, t1vixchain, (vx1expiryf - today).days, rfr, vx1close)
    values["varqt1"] = varqt1

    if pd.notna(ervart1) and pd.notna(varqt1):
        values["eqv1"] = np.sqrt(ervart1 - varqt1)

    print('calc efg2')
    for vx2expiry in [vx2expiryf, vx2expiryb]:
        t2 = (vx2expiry - today).days
        if vx2expiry in SPXexpiries:
            print('calc ervar:', vx2expiry)
            t2chain = SPXoptionchain[vx2expiry]
            ervart2 = EFG.calc_forward_iv(
                session, t2chain, t2, rfr, spot
            )
        else:
            if (
                SPXexpiries[0] > vx2expiry
            ):
                exp1 = today
            else:
                exp1 = max([d for d in SPXexpiries if d < vx2expiry], default=None)
            exp2 = min([d for d in SPXexpiries if d > vx2expiry], default=None)
            print('calc ervar:', exp1, exp2)
            t2e1 = (exp1 - today).days  # front expiry of this vix future
            t2e2 = (
                exp2 - today
            ).days
            if date == exp1:
                vart2e1 = 0
            else:
                t2e1chain = SPXoptionchain[exp1]
                vart2e1 = EFG.calc_forward_iv(
                    session, t2e1chain, t2e1, rfr, spot
                )
            t2e2chain = SPXoptionchain[exp2]
            vart2e2 = EFG.calc_forward_iv(
                session, t2e2chain, t2e2, rfr, spot
            )
            ervart2 = ((t2e2 - t2) * t2e1 * vart2e1 + (t2 - t2e1) * t2e2 * vart2e2) / (
                (t2e2 - t2e1) * t2
            )
        ervart2values.append((t2, ervart2))
    ervart2 = (
        ervart2values[1][0] * ervart2values[1][1]
        - ervart2values[0][0] * ervart2values[0][1]
    ) / (
        ervart2values[1][0] - ervart2values[0][0]
    )
    values["ervart2"] = ervart2

    assert vx2expiryf in VIXexpiries
    t2vixchain = VIXoptionchain[vx2expiryf]
    print('calc varq:', vx2expiryf)
    varqt2 = EFG.calc_varq(session, t2vixchain, (vx2expiryf - today).days, rfr, vx2close)
    values["varqt2"] = varqt2

    if pd.notna(ervart2) and pd.notna(varqt2):
        values["eqv2"] = np.sqrt(ervart2 - varqt2)

    efg1, efg2 = 100*(vx1close - values["eqv1"])/vx1close, 100*(vx2close - values["eqv2"])/vx2close
    print(values)
    print(f'SPX last price: {spot}')
    print(f'VX1 prev close: {VXfmdata[0].prev_close}, VX2 prev close: {VXbmdata[0].prev_close}')
    print(f'VX1 close: {VXfmdata[0].close}, VX2 close: {VXbmdata[0].close}')
    print(f'VX1 last price: {VXfmdata[0].last}, VX2 last price: {VXbmdata[0].last}')
    print(f'EFG1: {efg1}, EFG2: {efg2}')

    # =============================================================================
    # TELEGRAM INTEGRATION
    # =============================================================================

    formatted_values = []
    for key, value in values.items():
        # Convert np.float64 to regular float and round to 4 decimal places
        formatted_value = round(float(value), 4)
        formatted_values.append(f"{key}: {formatted_value}")

    values_display = " || ".join(formatted_values)

    # Prepare the message with all the printed information
    telegram_message = f"""📊 <b>EFG Results</b>
    🕐 <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>

    <b>Values:</b>
    {values_display}

    <b>Market Data:</b>
    SPX last price: {spot}
    RFR: {rfr}

    <b>VX1 Data:</b>
    {vx1expiryf} - {vx1expiryb}
    • Prev close: {VXfmdata[0].prev_close}
    • Close: {VXfmdata[0].close}
    • Last price: {VXfmdata[0].last}

    <b>VX2 Data:</b>
    {vx2expiryf} - {vx2expiryb}
    • Prev close: {VXbmdata[0].prev_close}
    • Close: {VXbmdata[0].close}
    • Last price: {VXbmdata[0].last}

    <b>🎯 Final Results:</b>
    • EFG1: {efg1:.4f}
    • EFG2: {efg2:.4f}
    """
    send_to_telegram(telegram_message)
    return

# =============================================================================
# scraping
def ldm(d): nd = datetime(d.year + 1, 1, 1) if d.month == 12 else datetime(d.year, d.month + 1,
                                                                               1); return nd - timedelta(days=1)
def scrape():
    SPXoptionchain = get_option_chain(session, "SPX")
    VIXoptionchain = get_option_chain(session, "VIX")
    try:
        treasury_bill = yf.Ticker("^IRX", session=yfsession)
        hist = treasury_bill.history(period="1d")
        rfr = float(hist['Close'].iloc[-1])
    except Exception as e:
        rfr = 4.5

    SPXexpiries = list(SPXoptionchain.keys())
    VIXexpiries = list(VIXoptionchain.keys())

    SPXdata = get_market_data_by_type(session, indices=['SPX'])
    VIXdata = get_market_data_by_type(session, indices=['VIX'])
    first5 = SPXexpiries[:5]
    sd = first5[-1] + timedelta(days=1)
    eoms = [d for d in SPXexpiries if d >= sd and d.day == ldm(d).day][:6]
    SPXsavelist = first5 + eoms
    VIXsavelist = VIXexpiries[:10]
    SPXspot = float(SPXdata[0].last)
    VIXspot = float(VIXdata[0].last)

    payload = {
        "SPX": {"spot": SPXspot, "optionchain": {}},
        "VIX": {"spot": VIXspot, "optionchain": {}},
        "rfr": rfr
    }

    for exp in SPXsavelist:
        chain = [opt for opt in SPXoptionchain[exp] if opt.settlement_type == "PM"]
        df = OptionMethods.convertchain(session, chain)
        df = df.apply(pd.to_numeric).replace([np.nan, np.inf, -np.inf], None)
        df = df.to_dict(orient="records")
        payload["SPX"]["optionchain"][exp.isoformat()] = df

    for exp in VIXsavelist:
        chain = VIXoptionchain[exp]
        df = OptionMethods.convertchain(session, chain)
        df = df.apply(pd.to_numeric).replace([np.nan, np.inf, -np.inf], None)
        df = df.to_dict(orient="records")
        payload["VIX"]["optionchain"][exp.isoformat()] = df

    snapshot = {
        "date": date.today().isoformat(),
        "period": "AM" if nytime.time() < time(12, 0) else "PM",
        "fetchtime": nytime.isoformat(),
        "payload": payload
    }

    PROJECT_URL = config.database_url
    TABLE_NAME = "market_snapshots"
    API_KEY = config.database_service_key

    headers = {
        "apikey": API_KEY,
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    url = f"{PROJECT_URL}/rest/v1/{TABLE_NAME}"

    try:
        response = requests.post(url, headers=headers, json=snapshot)
        if response.status_code != 201 and response.status_code != 200:
            send_to_telegram(f"Status {response.status_code} {response.json()}")
        else:
            send_to_telegram(f"Data sent to db successfully at {snapshot["fetchtime"]}")
    except Exception as e:
        telegram_message = f"Error sending data to db: {e}"
        send_to_telegram(telegram_message)


market_session = get_market_sessions(session, exchanges=[ExchangeType.NYSE])[0].status
if market_session != MarketStatus.CLOSED:
    efg()
    scrape()


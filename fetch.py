"""Daily market-data snapshot for the BTC probability dashboard.

Runs on GitHub Actions (which has open internet access) and writes:
  data/btc_daily.csv    date, price, mvrv, realized_price            (Coin Metrics community API)
  data/macro_daily.csv  date, us10y, real10y, usd_broad             (FRED, from 2010)
  data/latest.json      latest values the dashboard needs, each with its date and source

Price, MVRV, realized price, the 200-day and 200-week averages and the all-time high all come
from the same Coin Metrics daily series the model was trained on, so live features match training.
Only the standard library is used. Every source is optional except Coin Metrics: a failed source
is written as null and listed under "errors", and the run still succeeds.
"""
import csv, json, os, sys, time, datetime as dt
import urllib.request, urllib.error

OUT = os.environ.get("OUT_DIR", "data")
UA = {"User-Agent": "btc-dashboard-data/1.0 (+github actions)"}
CM = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"


def get(url, retries=3):
    """HTTP GET returning text; tests replace this function."""
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def num(v):
    try:
        x = float(v)
        return x if x == x else None  # drop NaN
    except (TypeError, ValueError):
        return None


def coinmetrics():
    """Full daily history of price and MVRV (both in the free community tier); realized price = price / MVRV."""
    url = (f"{CM}?assets=btc&metrics=PriceUSD,CapMVRVCur&frequency=1d"
           f"&start_time=2010-07-18&page_size=10000")
    rows = []
    while url:
        page = json.loads(get(url))
        for r in page.get("data", []):
            price, mvrv = num(r.get("PriceUSD")), num(r.get("CapMVRVCur"))
            rows.append({"date": r["time"][:10], "price": price, "mvrv": mvrv,
                         "realized_price": price / mvrv if price and mvrv else None})
        url = page.get("next_page_url")
    rows = [r for r in rows if r["price"]]
    rows.sort(key=lambda r: r["date"])
    if len(rows) < 1400:
        raise RuntimeError(f"Coin Metrics returned only {len(rows)} days")
    return rows


def fear_greed():
    d = json.loads(get("https://api.alternative.me/fng/?limit=1"))["data"][0]
    day = dt.datetime.fromtimestamp(int(d["timestamp"]), dt.timezone.utc).date().isoformat()
    return num(d["value"]), day


def fred(series):
    """FRED series from 2010 as {date: value}; '.' marks a missing day."""
    text = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2010-01-01")
    out = {}
    for row in csv.DictReader(text.splitlines()):
        date = row.get("observation_date") or row.get("DATE")
        v = num(row.get(series))
        if date and v is not None:
            out[date] = v
    if not out:
        raise RuntimeError(f"FRED {series}: no rows")
    return out


def dxy():
    j = json.loads(get("https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?range=5d&interval=1d"))
    meta = j["chart"]["result"][0]["meta"]
    day = dt.datetime.fromtimestamp(int(meta["regularMarketTime"]), dt.timezone.utc).date().isoformat()
    return num(meta["regularMarketPrice"]), day


def funding():
    """8-hour BTC perpetual funding in percent: OKX first, Deribit as fallback."""
    try:
        d = json.loads(get("https://www.okx.com/api/v5/public/funding-rate?instId=BTC-USDT-SWAP"))["data"][0]
        return num(d["fundingRate"]) * 100, "OKX BTC-USDT-SWAP"
    except Exception:
        d = json.loads(get("https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-PERPETUAL"))["result"]
        return num(d["funding_8h"]) * 100, "Deribit BTC-PERPETUAL"


def main():
    os.makedirs(OUT, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    latest = {"generated_at": now, "values": {}, "errors": []}
    v = latest["values"]

    def put(key, value, as_of, source):
        v[key] = {"value": None if value is None else round(value, 6), "as_of": as_of, "source": source}

    # Core: Coin Metrics (the run fails without it)
    cm = coinmetrics()
    with open(f"{OUT}/btc_daily.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "price", "mvrv", "realized_price"])
        w.writeheader()
        for r in cm:
            w.writerow({k: ("" if r[k] is None else (round(r[k], 6) if k != "date" else r[k])) for k in w.fieldnames})
    last = cm[-1]
    age = (dt.date.fromisoformat(now[:10]) - dt.date.fromisoformat(last["date"])).days
    if age > 3:
        latest["errors"].append(f"coinmetrics: data is stale, last day {last['date']} ({age} days old)")
    prices = [r["price"] for r in cm]
    src = "Coin Metrics community API (PriceUSD, CapMVRVCur)"
    put("price", last["price"], last["date"], src + ", дневное закрытие UTC")
    put("realized_price", last["realized_price"], last["date"], src + ", цена / MVRV")
    put("mvrv", last["mvrv"], last["date"], src)
    put("ma200d", sum(prices[-200:]) / 200, last["date"], src + ", 200 дневных закрытий")
    put("ma200w", sum(prices[-1400:]) / 1400, last["date"], src + ", 1400 дневных закрытий")
    i_ath = max(range(len(cm)), key=lambda i: cm[i]["price"])
    put("ath", cm[i_ath]["price"], cm[i_ath]["date"], src + ", максимальное дневное закрытие")

    # Optional sources
    def optional(name, fn):
        try:
            fn()
        except Exception as e:
            latest["errors"].append(f"{name}: {e}")

    def _fng():
        val, day = fear_greed()
        put("fear_greed", val, day, "alternative.me Crypto Fear & Greed")

    def _macro():
        s = {"us10y": fred("DGS10"), "real10y": fred("DFII10"), "usd_broad": fred("DTWEXBGS")}
        dates = sorted(set().union(*[d.keys() for d in s.values()]))
        with open(f"{OUT}/macro_daily.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "us10y", "real10y", "usd_broad"])
            for day in dates:
                w.writerow([day] + [s[k].get(day, "") for k in ("us10y", "real10y", "usd_broad")])
        for k, name in (("us10y", "FRED DGS10"), ("real10y", "FRED DFII10")):
            day = max(s[k])
            put(k, s[k][day], day, name)

    def _dxy():
        val, day = dxy()
        put("dxy", val, day, "Yahoo Finance DX-Y.NYB")

    def _funding():
        val, name = funding()
        put("funding_pct", val, now[:10], name + ", ставка за 8 ч")

    optional("fear_greed", _fng)
    optional("fred", _macro)
    optional("dxy", _dxy)
    optional("funding", _funding)

    with open(f"{OUT}/latest.json", "w") as f:
        json.dump(latest, f, ensure_ascii=False, indent=2)
    print(json.dumps(latest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())

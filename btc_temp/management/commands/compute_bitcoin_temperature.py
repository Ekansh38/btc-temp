from django.core.management.base import BaseCommand
from django.db import transaction
from btc_temp.models import BitcoinTemperature, MvrvZScoreDaily

import os, sys, re
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, date, timedelta
from math import erf, sqrt

BASE_URL = "https://api.bitcoinmagazinepro.com/metrics"
DEFAULT_METRIC = "mvrv-zscore"
STALE_DAYS = 3          # if latest stored date is older than this, do a backfill
BACKFILL_WINDOW = 14    # days to backfill when stale

# -------- CSV cleaner you already validated --------
NUM_OR_NA = r'(?:[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|NaN|nan|NAN|inf|-inf|)'
WS = r'\s*'
Q = r'"?'

ROW_RE = re.compile(
    rf'^{WS}\d+{WS},{WS}'
    rf'{Q}\d{{4}}-\d{{2}}-\d{{2}}{Q}{WS},{WS}'
    rf'{Q}{NUM_OR_NA}{Q}{WS},{WS}'
    rf'{Q}{NUM_OR_NA}{Q}{WS},{WS}'
    rf'{Q}{NUM_OR_NA}{Q}{WS},{WS}'
    rf'{Q}{NUM_OR_NA}{Q}{WS}$'
)
HEADER = ",Date,Price,MarketCap,realized_cap,ZScore"

def _sanitize_line(line: str) -> str:
    line = line.replace("\x00", "")
    line = "".join(ch for ch in line if ch == "\t" or ord(ch) >= 9)
    return line.strip()

def clean_and_parse_csv_text(raw: str) -> pd.DataFrame:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")

    lines = [_sanitize_line(l) for l in text.split("\n")]
    lines = [l for l in lines if l]
    data_lines = [l for l in lines if ROW_RE.match(l)]

    if not data_lines:
        df = pd.read_csv(
            StringIO(text),
            names=["Index", "Date", "Price", "MarketCap", "realized_cap", "ZScore"],
            usecols=[0,1,2,3,4,5],
            engine="python",
            on_bad_lines="skip",
            skip_blank_lines=True,
        )
        if df.empty:
            sample = "\n".join(lines[:100])
            raise ValueError("No valid rows; first 100 sanitized lines:\n" + sample)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
        for col in ["Index","Price","MarketCap","realized_cap","ZScore"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    cleaned = HEADER + "\n" + "\n".join(data_lines)
    df = pd.read_csv(
        StringIO(cleaned),
        dtype={"Price":"float64","MarketCap":"float64","realized_cap":"float64","ZScore":"float64"},
        na_values=["NaN","nan","NAN",""]
    )
    first_col = [int(l.split(",", 1)[0]) for l in data_lines]
    df.insert(0, "Index", first_col)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    return df

# -------- helpers --------
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def last_6y_cutoff() -> date:
    return (datetime.utcnow().date() - timedelta(days=6*365))

def fetch_metric(api_key: str, metric: str, from_date: str | None) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"from_date": from_date} if from_date else None
    r = requests.get(f"{BASE_URL}/{metric}", headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return clean_and_parse_csv_text(r.text)

def upsert_rows(df: pd.DataFrame) -> int:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(MvrvZScoreDaily(
            date=row.Date.date(),
            price=None if pd.isna(row.Price) else row.Price,
            market_cap=None if pd.isna(row.MarketCap) else row.MarketCap,
            realized_cap=None if pd.isna(row.realized_cap) else row.realized_cap,
            zscore=None if pd.isna(row.ZScore) else row.ZScore,
        ))
    if not rows:
        return 0
    with transaction.atomic():
        MvrvZScoreDaily.objects.bulk_create(rows, ignore_conflicts=True)
    return len(rows)

def weekly_reduce(df: pd.DataFrame) -> pd.DataFrame:
    # keep the latest row per ISO week
    df = df.copy()
    df["iso_year"] = df["Date"].dt.isocalendar().year
    df["iso_week"] = df["Date"].dt.isocalendar().week
    return df.sort_values("Date").groupby(["iso_year", "iso_week"], as_index=False).tail(1)

# -------- main command --------
class Command(BaseCommand):
    help = "Auto-update MVRV data if needed (delta + backfill), then compute/save 6y BTC Temperature."

    def add_arguments(self, parser):
        parser.add_argument("--metric", default=DEFAULT_METRIC)
        parser.add_argument("--store-weekly", action="store_true",
                            help="Store at most one row per ISO week (latest in that week).")
        parser.add_argument("--tail", type=int, default=5,
                            help="Print last N ingested series rows (default: 5).")

    def handle(self, *args, **opts):
        api_key = os.getenv("BM_PRO_API_KEY")
        if not api_key:
            self.stderr.write("Error: BM_PRO_API_KEY is not set")
            sys.exit(1)

        metric = opts["metric"]
        store_weekly = bool(opts["store_weekly"])
        tail_n = max(1, int(opts["tail"]))

        # 1) figure out where to start
        last = MvrvZScoreDaily.objects.order_by("-date").first()
        from_date = last.date.isoformat() if last else "2010-01-01"

        # 2) delta fetch new rows
        df_new = fetch_metric(api_key, metric, from_date)
        if last:
            df_new = df_new[df_new["Date"].dt.date > last.date]
        if store_weekly and not df_new.empty:
            df_new = weekly_reduce(df_new)

        added = upsert_rows(df_new)
        if added:
            self.stdout.write(f"Ingested {added} new {'weekly' if store_weekly else 'daily'} rows.")
            self.stdout.write(df_new[["Date","ZScore"]].sort_values("Date").tail(tail_n).to_string(index=False))
        else:
            self.stdout.write("No new rows from delta fetch.")

        # 3) automatic backfill if stale
        latest_db = MvrvZScoreDaily.objects.order_by("-date").values_list("date", flat=True).first()
        if latest_db and (date.today() - latest_db) > timedelta(days=STALE_DAYS):
            self.stdout.write(self.style.WARNING(
                f"Latest stored date {latest_db} is >{STALE_DAYS} days old. Backfilling last {BACKFILL_WINDOW} days…"
            ))
            back_from = (date.today() - timedelta(days=BACKFILL_WINDOW)).isoformat()
            df_back = fetch_metric(api_key, metric, back_from)
            # only rows newer than latest_db
            df_back = df_back[df_back["Date"].dt.date > latest_db]
            if store_weekly and not df_back.empty:
                df_back = weekly_reduce(df_back)
            added_back = upsert_rows(df_back)
            self.stdout.write(f"Backfill added {added_back} rows.")

        # 4) compute BTC temperature over last 6y from DB
        cutoff = last_6y_cutoff()
        qs = (MvrvZScoreDaily.objects
            .filter(date__gte=cutoff)
            .order_by("date")
            .values("date", "zscore"))

        df_6y = pd.DataFrame(list(qs))
        if df_6y.empty:
            self.stderr.write("No data in DB to compute temperature. Aborting.")
            sys.exit(2)

        # --- NEW: coerce Decimal -> float and drop NaNs ---
        df_6y["zscore"] = pd.to_numeric(df_6y["zscore"], errors="coerce")
        df_6y = df_6y.dropna(subset=["zscore"])
        # ---------------------------------------------------

        current = float(df_6y["zscore"].iloc[-1])
        mean_6y = float(df_6y["zscore"].mean())
        std_6y  = float(df_6y["zscore"].std(ddof=1))
        z = 0.0 if (std_6y == 0 or pd.isna(std_6y)) else (current - mean_6y) / std_6y
        btc_temp = norm_cdf(z) * 100.0


        # 5) save one temperature row
        with transaction.atomic():
            trow = BitcoinTemperature.objects.create(
                temperature=round(btc_temp, 6),
                inputs={
                    "metric": metric,
                    "window": "6y",
                    "latest_zscore": current,
                    "mean_6y": mean_6y,
                    "std_6y": None if pd.isna(std_6y) else std_6y,
                    "date_latest": str(df_6y["date"].iloc[-1]),
                    "store_mode": "weekly" if store_weekly else "daily",
                    "stale_threshold_days": STALE_DAYS,
                    "backfill_days": BACKFILL_WINDOW,
                },
                calc_version="stage1",
            )

        total_temps = BitcoinTemperature.objects.count()
        self.stdout.write(self.style.SUCCESS(
            f"Saved BTC Temp #{trow.id}: {btc_temp:.2f} "
            f"(z={current:.4f}, mean={mean_6y:.4f}, std={std_6y:.4f}) "
            f"[total temps: {total_temps}]"
        ))

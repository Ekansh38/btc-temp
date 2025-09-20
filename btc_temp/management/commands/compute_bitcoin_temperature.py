from django.core.management.base import BaseCommand
from django.db import transaction
from btc_temp.models import (
    BitcoinTemperature,
    MvrvZScoreDaily,
    RhodlRatioDaily,
    FundingRatesDaily,
)

import os, sys, re
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, date, timedelta
from math import erf, sqrt

BASE_URL = "https://api.bitcoinmagazinepro.com/metrics"
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

def fetch_metric_text(api_key: str, metric: str, from_date: str | None):
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"from_date": from_date} if from_date else None
    r = requests.get(f"{BASE_URL}/{metric}", headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.text

def fetch_metric_df(api_key: str, metric: str, from_date: str | None) -> pd.DataFrame:
    return clean_and_parse_csv_text(fetch_metric_text(api_key, metric, from_date))

def fetch_metric_df_generic(api_key: str, metric: str, from_date: str | None) -> pd.DataFrame:
    """Tolerant reader for metrics that are (Date, Value...) shaped (e.g., rhodl-ratio, funding).
       Does NOT assume the 6-column MVRV layout; avoids usecols."""
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"from_date": from_date} if from_date else None
    r = requests.get(f"{BASE_URL}/{metric}", headers=headers, params=params, timeout=30)
    r.raise_for_status()

    txt = r.text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\\n" in txt and "\n" not in txt:
        txt = txt.replace("\\n", "\n")
    if txt.startswith('",'):
        txt = txt[2:]

    # Tolerant parse: no fixed columns, skip malformed lines
    df = pd.read_csv(StringIO(txt), engine="python", on_bad_lines="skip")

    # Handle possible leading index column (common pattern)
    cols = [str(c) for c in df.columns]
    if len(cols) >= 2 and cols[1].lower() == "date" and cols[0].lower().startswith("unnamed"):
        df = df.iloc[:, 1:]  # drop the unnamed index col
        cols = [str(c) for c in df.columns]

    # Ensure there's a 'Date' column and parse it
    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    elif "Date" not in df.columns:
        # assume first column is Date if header isn't explicit
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def weekly_reduce(df: pd.DataFrame) -> pd.DataFrame:
    # keep the latest row per ISO week
    df = df.copy()
    df["iso_year"] = df["Date"].dt.isocalendar().year
    df["iso_week"] = df["Date"].dt.isocalendar().week
    return df.sort_values("Date").groupby(["iso_year", "iso_week"], as_index=False).tail(1)

# ---- Ingestors ----
def ingest_mvrv(api_key: str, store_weekly: bool) -> int:
    metric = "mvrv-zscore"
    last = MvrvZScoreDaily.objects.order_by("-date").first()
    df = fetch_metric_df_generic(api_key, metric, last.date.isoformat() if last else "2010-01-01")

    if last:
        df = df[df["Date"].dt.date > last.date]
    if store_weekly and not df.empty:
        df = weekly_reduce(df)

    rows = []
    for r in df.itertuples(index=False):
        rows.append(MvrvZScoreDaily(
            date=r.Date.date(),
            price=None if pd.isna(r.Price) else r.Price,
            market_cap=None if pd.isna(r.MarketCap) else r.MarketCap,
            realized_cap=None if pd.isna(r.realized_cap) else r.realized_cap,
            zscore=None if pd.isna(r.ZScore) else r.ZScore,
        ))
    if not rows:
        return 0
    with transaction.atomic():
        MvrvZScoreDaily.objects.bulk_create(rows, ignore_conflicts=True)
    return len(rows)

def _pick_value_column(df: pd.DataFrame) -> str:
    # Try common names first
    common = [c for c in df.columns if c.lower() in ("zscore","value","ratio","funding","funding_rate","fundingrates")]
    if common:
        return common[0]
    # Else take the rightmost numeric column that isn't Date/Index
    numeric_cols = [c for c in df.columns if c not in ("Index","Date")]
    # prefer columns that are numeric
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[-1] if numeric_cols else df.columns[-1]

def ingest_value_metric(api_key: str, metric: str, model, store_weekly: bool) -> int:
    """Generic ingestor for (Date, value) shape metrics (e.g., rhodl-ratio, bitcoin-funding-rates)."""
    last = model.objects.order_by("-date").first()
    df = fetch_metric_df_generic(api_key, metric, last.date.isoformat() if last else "2010-01-01")

    if last:
        df = df[df["Date"].dt.date > last.date]
    if store_weekly and not df.empty:
        df = weekly_reduce(df)

    if df.empty:
        return 0

    value_col = _pick_value_column(df)
    rows = []
    for r in df.itertuples(index=False):
        val = getattr(r, value_col)
        rows.append(model(
            date=r.Date.date(),
            value=None if pd.isna(val) else val
        ))
    with transaction.atomic():
        model.objects.bulk_create(rows, ignore_conflicts=True)
    return len(rows)

# ---- Temperature computation ----
def compute_temp_from_model(model, value_field: str):
    cutoff = last_6y_cutoff()
    qs = (model.objects
          .filter(date__gte=cutoff)
          .order_by("date")
          .values("date", value_field))
    df = pd.DataFrame(list(qs))
    if df.empty:
        qs = model.objects.all().order_by("date").values("date", value_field)
        df = pd.DataFrame(list(qs))
        if df.empty:
            raise RuntimeError(f"No data in {model.__name__} to compute temperature.")

    # Decimal -> float + drop NaNs
    df[value_field] = pd.to_numeric(df[value_field], errors="coerce")
    df = df.dropna(subset=[value_field])

    current = float(df[value_field].iloc[-1])
    mean = float(df[value_field].mean())
    std  = float(df[value_field].std(ddof=1))
    z = 0.0 if (std == 0 or pd.isna(std)) else (current - mean) / std
    temp = norm_cdf(z) * 100.0
    latest_date = str(df["date"].iloc[-1])
    return temp, current, mean, std, latest_date

# -------- main command --------
class Command(BaseCommand):
    help = (
        "Stage 2: ingest/update MVRV (70%), RHODL (15%), Funding (15%); "
        "compute each 6y temperature; save weighted Stage-2 temperature."
    )

    def add_arguments(self, parser):
        parser.add_argument("--store-weekly", action="store_true",
                            help="Store at most one row per ISO week (latest in that week) for all metrics.")
        parser.add_argument("--tail", type=int, default=5,
                            help="Print last N ingested rows preview per metric (default: 5).")

    def handle(self, *args, **opts):
        api_key = os.getenv("BM_PRO_API_KEY")
        if not api_key:
            self.stderr.write("Error: BM_PRO_API_KEY is not set")
            sys.exit(1)

        store_weekly = bool(opts["store_weekly"])
        tail_n = max(1, int(opts["tail"]))

        # 1) Ingest deltas for all 3 metrics
        added_mvrv = ingest_mvrv(api_key, store_weekly)
        added_rhodl = ingest_value_metric(api_key, "rhodl-ratio", RhodlRatioDaily, store_weekly)
        added_fund  = ingest_value_metric(api_key, "fr-average", FundingRatesDaily, store_weekly)


        self.stdout.write(f"Ingested — MVRV: {added_mvrv} | RHODL: {added_rhodl} | Funding: {added_fund}")

        # Tiny tail previews (safe if nothing added)
        if added_mvrv:
            qs = MvrvZScoreDaily.objects.order_by("date").values("date","zscore")
            dfp = pd.DataFrame(list(qs)).tail(tail_n); dfp["zscore"]=pd.to_numeric(dfp["zscore"], errors="coerce")
            self.stdout.write("MVRV tail:\n" + dfp.to_string(index=False))
        if added_rhodl:
            qs = RhodlRatioDaily.objects.order_by("date").values("date","value")
            dfp = pd.DataFrame(list(qs)).tail(tail_n); dfp["value"]=pd.to_numeric(dfp["value"], errors="coerce")
            self.stdout.write("RHODL tail:\n" + dfp.to_string(index=False))
        if added_fund:
            qs = FundingRatesDaily.objects.order_by("date").values("date","value")
            dfp = pd.DataFrame(list(qs)).tail(tail_n); dfp["value"]=pd.to_numeric(dfp["value"], errors="coerce")
            self.stdout.write("Funding tail:\n" + dfp.to_string(index=False))

        # 2) Compute per-metric temps (6y)
        mvrv_temp, mvrv_cur, mvrv_mean, mvrv_std, mvrv_date = compute_temp_from_model(MvrvZScoreDaily, "zscore")
        rhodl_temp, rhodl_cur, rhodl_mean, rhodl_std, rhodl_date = compute_temp_from_model(RhodlRatioDaily, "value")
        fund_temp,  fund_cur,  fund_mean,  fund_std,  fund_date  = compute_temp_from_model(FundingRatesDaily, "value")

        # 3) Weighted combine: 70/15/15
        w_m, w_r, w_f = 0.70, 0.15, 0.15
        stage2_temp = (w_m * mvrv_temp) + (w_r * rhodl_temp) + (w_f * fund_temp)

        # 4) Save one temperature row (Stage 2)
        with transaction.atomic():
            obj = BitcoinTemperature.objects.create(
                temperature=round(stage2_temp, 6),
                inputs={
                    "stage": "2",
                    "weights": {"mvrv": w_m, "rhodl": w_r, "funding": w_f},
                    "components": {
                        "mvrv":   {"temp": mvrv_temp, "current": mvrv_cur, "mean_6y": mvrv_mean, "std_6y": mvrv_std, "latest": mvrv_date},
                        "rhodl":  {"temp": rhodl_temp, "current": rhodl_cur, "mean_6y": rhodl_mean, "std_6y": rhodl_std, "latest": rhodl_date},
                        "funding":{"temp": fund_temp,  "current": fund_cur,  "mean_6y": fund_mean,  "std_6y": fund_std,  "latest": fund_date},
                    },
                    "window": "6y",
                },
                calc_version="stage2",
            )

        # 5) Clear, friendly output showing exactly what's happening
        self.stdout.write("\n=== Stage 2 Bitcoin Temperature (Weighted) ===")
        self.stdout.write(f"MVRV (70%):   latest={mvrv_cur:.6f}  temp={mvrv_temp:.2f}  (mean_6y={mvrv_mean:.6f}, std_6y={mvrv_std:.6f}, latest_date={mvrv_date})")
        self.stdout.write(f"RHODL (15%):  latest={rhodl_cur:.6f} temp={rhodl_temp:.2f} (mean_6y={rhodl_mean:.6f}, std_6y={rhodl_std:.6f}, latest_date={rhodl_date})")
        self.stdout.write(f"Funding (15%):latest={fund_cur:.6f}  temp={fund_temp:.2f}  (mean_6y={fund_mean:.6f}, std_6y={fund_std:.6f}, latest_date={fund_date})")
        self.stdout.write(self.style.SUCCESS(f"\n=> Final Stage-2 Temperature: {stage2_temp:.2f}"))
        total = BitcoinTemperature.objects.count()
        self.stdout.write(self.style.SUCCESS(f"Saved as row #{obj.id} (calc_version=stage2). Total temp rows: {total}"))

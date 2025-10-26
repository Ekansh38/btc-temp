from typing import Optional
from django.core.management.base import BaseCommand
from django.db import transaction
from btc_temp.models import (
    BitcoinTemperature,
    MvrvZScoreDaily,
    RhodlRatioDaily,
    ShillerPeDaily,


)
# put these near your imports
import json
from datetime import datetime, date
from decimal import Decimal

def safe_json(obj):
    """Convert Decimals/dates to JSON-safe primitives."""
    def convert(o):
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return o
    # walk the structure and convert
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return convert(x)
    return walk(obj)

import os, sys, re
import requests
import pandas as pd
from io import StringIO
from datetime import datetime, date, timedelta, timezone
from math import erf, sqrt

# ------------------------------
# CONFIG
# ------------------------------
BASE_URL = "https://api.bitcoinmagazinepro.com/metrics"
STALE_DAYS = 3
BACKFILL_WINDOW = 14

# ------------------------------
# CSV CLEANER for MVRV
# ------------------------------
NUM_OR_NA = r"(?:[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|NaN|nan|NAN|inf|-inf|)"
WS = r"\s*"
Q = r'"?'
ROW_RE = re.compile(
    rf"^{WS}\d+{WS},{WS}"
    rf"{Q}\d{{4}}-\d{{2}}-\d{{2}}{Q}{WS},{WS}"
    rf"{Q}{NUM_OR_NA}{Q}{WS},{WS}"
    rf"{Q}{NUM_OR_NA}{Q}{WS},{WS}"
    rf"{Q}{NUM_OR_NA}{Q}{WS},{WS}"
    rf"{Q}{NUM_OR_NA}{Q}{WS}$"
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
        df = pd.read_cscv(
            StringIO(text),
            names=["Index", "Date", "Price", "MarketCap", "realized_cap", "ZScore"],
            usecols=[0, 1, 2, 3, 4, 5],
            engine="python",
            on_bad_lines="skip",
            skip_blank_lines=True,
        )
        if df.empty:
            sample = "\n".join(lines[:100])
            raise ValueError("No valid rows; first 100 sanitized lines:\n" + sample)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df = df.dropna(subset=["Date"]).reset_index(drop=True)
        for col in ["Index", "Price", "MarketCap", "realized_cap", "ZScore"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    cleaned = HEADER + "\n" + "\n".join(data_lines)
    df = pd.read_csv(
        StringIO(cleaned),
        dtype={
            "Price": "float64",
            "MarketCap": "float64",
            "realized_cap": "float64",
            "ZScore": "float64",
        },
        na_values=["NaN", "nan", "NAN", ""],
    )
    first_col = [int(l.split(",", 1)[0]) for l in data_lines]
    df.insert(0, "Index", first_col)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    return df


# ------------------------------
# HELPERS
# ------------------------------
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def last_6y_cutoff() -> date:
    return datetime.utcnow().date() - timedelta(days=6 * 365)


def fetch_metric_text(api_key: str, metric: str, from_date: Optional[str]):
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"from_date": from_date} if from_date else None
    r = requests.get(f"{BASE_URL}/{metric}", headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_metric_df_generic(api_key: str, metric: str, from_date: Optional[str]) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"from_date": from_date} if from_date else None
    r = requests.get(f"{BASE_URL}/{metric}", headers=headers, params=params, timeout=30)
    r.raise_for_status()

    txt = r.text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\\n" in txt and "\n" not in txt:
        txt = txt.replace("\\n", "\n")
    if txt.startswith('",'):
        txt = txt[2:]

    df = pd.read_csv(StringIO(txt), engine="python", on_bad_lines="skip")

    cols = [str(c) for c in df.columns]
    if (
        len(cols) >= 2
        and cols[1].lower() == "date"
        and cols[0].lower().startswith("unnamed")
    ):
        df = df.iloc[:, 1:]
        cols = [str(c) for c in df.columns]

    if "Date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    elif "Date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def weekly_reduce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["iso_year"] = df["Date"].dt.isocalendar().year
    df["iso_week"] = df["Date"].dt.isocalendar().week
    return df.sort_values("Date").groupby(["iso_year", "iso_week"], as_index=False).tail(1)


# ------------------------------
# INGESTORS (MVRV + RHODL)
# ------------------------------
def ingest_mvrv(api_key: str, store_weekly: bool) -> int:
    metric = "mvrv-zscore"
    last = MvrvZScoreDaily.objects.order_by("-date").first()
    df = fetch_metric_df_generic(
        api_key, metric, last.date.isoformat() if last else "2010-01-01"
    )

    if last:
        df = df[df["Date"].dt.date > last.date]
    if store_weekly and not df.empty:
        df = weekly_reduce(df)

    rows = []
    for r in df.itertuples(index=False):
        rows.append(
            MvrvZScoreDaily(
                date=r.Date.date(),
                price=getattr(r, "Price", None),
                market_cap=getattr(r, "MarketCap", None),
                realized_cap=getattr(r, "realized_cap", None),
                zscore=getattr(r, "ZScore", None),
            )
        )
    if not rows:
        return 0
    with transaction.atomic():
        MvrvZScoreDaily.objects.bulk_create(rows, ignore_conflicts=True)
    return len(rows)


def _pick_value_column(df: pd.DataFrame) -> str:
    common = [c for c in df.columns if c.lower() in ("zscore", "value", "ratio")]
    if common:
        return common[0]
    numeric_cols = [c for c in df.columns if c not in ("Index", "Date")]
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols[-1] if numeric_cols else df.columns[-1]


def ingest_value_metric(api_key: str, metric: str, model, store_weekly: bool) -> int:
    last = model.objects.order_by("-date").first()
    df = fetch_metric_df_generic(
        api_key, metric, last.date.isoformat() if last else "2010-01-01"
    )

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
        rows.append(model(date=r.Date.date(), value=None if pd.isna(val) else val))
    with transaction.atomic():
        model.objects.bulk_create(rows, ignore_conflicts=True)
    return len(rows)


# ------------------------------
# SHILLER PE SCRAPER + CSV INGEST
# ------------------------------
_MULTPL_URL = "https://www.multpl.com/shiller-pe"
_META_RE = re.compile(r'Current\s+Shiller\s+PE\s+Ratio\s+is\s+([0-9]+(?:\.[0-9]+)?)', re.IGNORECASE)
_CURRENT_BLOCK_RE = re.compile(
    r'id="current".*?Current.*?Shiller\s*PE\s*Ratio.*?:\s*([0-9]+(?:\.[0-9]+)?)',
    re.IGNORECASE | re.DOTALL
)
_PI_RE = re.compile(r'\bpi\s*=\s*\[\s*\[(.*?)\]\s*,\s*\[(.*?)\]', re.DOTALL)


def _http_get(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; btc-temp-bot/1.0)",
        "Accept": "text/html,application/xhtml+xml",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def _parse_number(s: str) -> Optional[float]:
    for regex in (_META_RE, _CURRENT_BLOCK_RE):
        m = regex.search(s)
        if m:
            try:
                return float(m.group(1))
            except:
                pass

    # fallback: from pi arrays
    m = _PI_RE.search(s)
    if m:
        try:
            vals = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', m.group(2))]
            if vals:
                return float(vals[-1])
        except:
            pass
    return None


def fetch_shiller_pe_current() -> tuple[datetime, float]:
    html = _http_get(_MULTPL_URL)
    val = _parse_number(html)
    if val is None:
        raise RuntimeError("Could not parse Shiller PE from multpl.com")
    return datetime.now(timezone.utc), val


def ingest_shiller_pe() -> int:
    as_of, value = fetch_shiller_pe_current()
    day = as_of.date()
    if ShillerPeDaily.objects.filter(date=day).exists():
        return 0
    ShillerPeDaily.objects.create(date=day, value=value)
    return 1


def ingest_shiller_from_csv(path: str) -> int:
    df = pd.read_csv(path)
    df = df.rename(columns={c.strip(): c.strip() for c in df.columns})
    if "Date" not in df.columns or "Value" not in df.columns:
        raise RuntimeError("Shiller CSV must have 'Date' and 'Value' columns")

    df["Date"] = df["Date"].astype(str).str.replace(".", "-", regex=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Date", "Value"]).sort_values("Date").reset_index(drop=True)

    rows = [ShillerPeDaily(date=r.Date.date(), value=r.Value) for r in df.itertuples(index=False)]
    if not rows:
        return 0
    with transaction.atomic():
        ShillerPeDaily.objects.bulk_create(rows, ignore_conflicts=True)
    return len(rows)


# ------------------------------
# TEMPERATURE CALCULATION
# ------------------------------
def compute_temp_from_model(model, value_field: str, use_full_history: bool = False):
    if not use_full_history:
        cutoff = last_6y_cutoff()
        qs = model.objects.filter(date__gte=cutoff).order_by("date").values("date", value_field)
    else:
        qs = model.objects.all().order_by("date").values("date", value_field)

    df = pd.DataFrame(list(qs))
    if df.empty:
        raise RuntimeError(f"No data in {model.__name__} to compute temperature.")

    df[value_field] = pd.to_numeric(df[value_field], errors="coerce")
    df = df.dropna(subset=[value_field])

    current = float(df[value_field].iloc[-1])
    mean = float(df[value_field].mean())
    std = float(df[value_field].std(ddof=1))
    z = 0.0 if (std == 0 or pd.isna(std)) else (current - mean) / std
    temp = norm_cdf(z) * 100.0
    latest_date = str(df["date"].iloc[-1])
    return temp, current, mean, std, latest_date


# ------------------------------
# MAIN COMMAND
# ------------------------------
class Command(BaseCommand):
    help = "Stage 2: ingest/update MVRV (70%), RHODL (15%), Shiller (15%); compute temperature."

    def add_arguments(self, parser):
        parser.add_argument(
            "--store-weekly",
            action="store_true",
            help="Store at most one row per ISO week (latest in that week).",
        )
        parser.add_argument(
            "--tail",
            type=int,
            default=5,
            help="Print last N ingested rows preview (default: 5).",
        )
        parser.add_argument(
            "--shiller-csv",
            type=str,
            help="Optional path to a Shiller CSV for initial ingestion (only needed once).",
        )

    def handle(self, *args, **opts):
        api_key = os.getenv("BM_PRO_API_KEY")
        if not api_key:
            self.stderr.write("Error: BM_PRO_API_KEY is not set")
            sys.exit(1)

        store_weekly = bool(opts["store_weekly"])
        tail_n = max(1, int(opts["tail"]))
        shiller_csv = opts.get("shiller_csv")

        # Initial Shiller CSV ingest if provided
        if shiller_csv:
            added_shiller_csv = ingest_shiller_from_csv(shiller_csv)
            self.stdout.write(f"Shiller CSV ingested: {added_shiller_csv} rows")

        # Regular ingest
        added_mvrv = ingest_mvrv(api_key, store_weekly)
        added_rhodl = ingest_value_metric(api_key, "rhodl-ratio", RhodlRatioDaily, store_weekly)
        added_shiller = ingest_shiller_pe()

        self.stdout.write(f"Ingested — MVRV: {added_mvrv} | RHODL: {added_rhodl} | Shiller: {added_shiller}")

        # Tail previews
        if added_mvrv:
            dfp = pd.DataFrame(list(MvrvZScoreDaily.objects.order_by("date").values("date", "zscore"))).tail(tail_n)
            self.stdout.write("MVRV tail:\n" + dfp.to_string(index=False))
        if added_rhodl:
            dfp = pd.DataFrame(list(RhodlRatioDaily.objects.order_by("date").values("date", "value"))).tail(tail_n)
            self.stdout.write("RHODL tail:\n" + dfp.to_string(index=False))
        if added_shiller:
            dfp = pd.DataFrame(list(ShillerPeDaily.objects.order_by("date").values("date", "value"))).tail(tail_n)
            self.stdout.write("Shiller tail:\n" + dfp.to_string(index=False))

        # Compute temps
        mvrv_temp, mvrv_cur, mvrv_mean, mvrv_std, mvrv_date = compute_temp_from_model(MvrvZScoreDaily, "zscore")
        rhodl_temp, rhodl_cur, rhodl_mean, rhodl_std, rhodl_date = compute_temp_from_model(RhodlRatioDaily, "value")
        shiller_temp, shiller_cur, shiller_mean, shiller_std, shiller_date = compute_temp_from_model(
            ShillerPeDaily, "value", use_full_history=True
        )

        # Weighting: 70/15/15
        w_m, w_r, w_s = 0.70, 0.15, 0.15
        stage2_temp = (w_m * mvrv_temp) + (w_r * rhodl_temp) + (w_s * shiller_temp)

        # Save to DB
        with transaction.atomic():
            inputs_data = {
                "stage": "2",
                "weights": {"mvrv": float(w_m), "rhodl": float(w_r)},
                "components": {
                    "mvrv": {
                        "temp": float(mvrv_temp),
                        "current": float(mvrv_cur),
                        "mean_6y": float(mvrv_mean),
                        "std_6y": float(mvrv_std),
                        "latest": str(mvrv_date),
                    },
                    "rhodl": {
                        "temp": float(rhodl_temp),
                        "current": float(rhodl_cur),
                        "mean_6y": float(rhodl_mean),
                        "std_6y": float(rhodl_std),
                        "latest": str(rhodl_date),
                    },
                },
                "window": "6y",
            }

            obj = BitcoinTemperature.objects.create(
                temperature=round(float(stage2_temp), 6),
                inputs=safe_json(inputs_data),
                calc_version="stage2",
            )


        self.stdout.write("\n=== Stage 2 Bitcoin Temperature (Weighted) ===")
        self.stdout.write(f"MVRV (70%):   latest={mvrv_cur:.6f}  temp={mvrv_temp:.2f}  (mean_6y={mvrv_mean:.6f}, std_6y={mvrv_std:.6f}, latest_date={mvrv_date})")
        self.stdout.write(f"RHODL (15%):  latest={rhodl_cur:.6f} temp={rhodl_temp:.2f} (mean_6y={rhodl_mean:.6f}, std_6y={rhodl_std:.6f}, latest_date={rhodl_date})")
        self.stdout.write(f"Shiller(15%): latest={shiller_cur:.6f}  temp={shiller_temp:.2f}  (mean_all={shiller_mean:.6f}, std_all={shiller_std:.6f}, latest_date={shiller_date})")
        self.stdout.write(self.style.SUCCESS(f"\n=> Final Stage-2 Temperature: {stage2_temp:.2f}"))


        # Output summary (unchanged below)

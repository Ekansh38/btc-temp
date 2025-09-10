from django.core.management.base import BaseCommand
import os, sys, re
import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from math import erf, sqrt

BASE_URL = "https://api.bitcoinmagazinepro.com/metrics"
DEFAULT_METRIC = "mvrv-zscore"

# Accept floats/ints/sci + NaNs/blanks, optional quotes/whitespace
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
    # Normalize newlines, strip outer whitespace
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()

    # If the API ever returns JSON-escaped text (with literal "\n"), unescape it:
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")

    lines = [_sanitize_line(l) for l in text.split("\n")]
    lines = [l for l in lines if l]

    # Keep only well-formed data rows
    data_lines = [l for l in lines if ROW_RE.match(l)]

    # Fallback: try lenient pandas read if regex matched nothing
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

    # Build cleaned CSV and parse
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

# Spreadsheet-equivalent helpers
def norm_cdf(z: float) -> float:
    # =NORM.DIST(z, 0, 1, TRUE)
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

class Command(BaseCommand):
    help = "Fetch BM Pro metric CSV, clean, and print summary + Bitcoin Temperature (last 6 years)."

    def add_arguments(self, parser):
        parser.add_argument("--metric", default=DEFAULT_METRIC)
        parser.add_argument("--tail", type=int, default=10, help="How many last rows to print (default: 10)")

    def handle(self, *args, **opts):
        api_key = os.getenv("BM_PRO_API_KEY")
        if not api_key:
            self.stderr.write("Error: BM_PRO_API_KEY is not set")
            sys.exit(1)

        metric = opts["metric"]
        url = f"{BASE_URL}/{metric}"
        headers = {"Authorization": f"Bearer {api_key}"}

        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        csv_text = r.text  # DO NOT print this!

        df = clean_and_parse_csv_text(csv_text)

        # ---- summary preview ----
        n = len(df)
        dmin = df["Date"].min()
        dmax = df["Date"].max()
        tail_n = max(1, int(opts["tail"]))
        self.stdout.write(f"Metric: {metric}")
        self.stdout.write(f"Rows: {n} | Date range: {dmin.date()} → {dmax.date()}")
        self.stdout.write(f"Last {tail_n} rows:")
        self.stdout.write(df.tail(tail_n).to_string(index=False))

        # ---------- Bitcoin Temperature over LAST 6 YEARS ----------
        # Window: last 6 years from "now" (rolling), not YTD
        cutoff = pd.Timestamp.utcnow() - pd.DateOffset(years=6)
        df_6y = df[df["Date"] >= cutoff]
        if df_6y.empty:
            # fallback: if somehow empty, use all data
            df_6y = df

        current = float(df["ZScore"].iloc[-1])
        mean_6y = float(df_6y["ZScore"].mean())
        std_6y = float(df_6y["ZScore"].std(ddof=1))  # sample std (= STDEV)

        z = 0.0 if std_6y == 0 or pd.isna(std_6y) else (current - mean_6y) / std_6y
        btc_temperature = norm_cdf(z) * 100.0

        self.stdout.write(
            f"\nBitcoin Temperature (Stage 1, last 6 years): {btc_temperature:.2f}"
            f"\n  current ZScore = {current:.6f}"
            f"\n  mean_6y        = {mean_6y:.6f}"
            f"\n  std_6y         = {std_6y:.6f}"
            f"\n  cutoff (UTC)   = {cutoff.date()}"
        )
        # -----------------------------------------------------------

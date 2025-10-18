# btc_temp/management/commands/ingest_shiller_csv.py
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.models import F
from btc_temp.models import ShillerPeDaily

import csv
from pathlib import Path
from datetime import datetime
from decimal import Decimal, InvalidOperation

import pandas as pd


def _parse_date(cell: str):
    """
    Accepts:
      - 'YYYY.MM' (e.g., '1881.01')
      - 'YYYY-MM'
      - 'YYYY/MM'
      - 'YYYY-MM-DD' / 'YYYY.MM.DD' / 'YYYY/MM/DD'
    Normalizes to first-of-month if day missing.
    Returns a date (naive) or raises ValueError.
    """
    s = (cell or "").strip()
    if not s:
        raise ValueError("empty date")

    # Normalize separators to '-'
    s = s.replace("/", "-").replace(".", "-")

    # If just year-month, add day 01
    parts = s.split("-")
    if len(parts) == 2:
        s = f"{parts[0]}-{parts[1]}-01"

    # Try a few formats
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y-%m-%d "):
        try:
            dt = datetime.strptime(s, fmt if fmt != "%Y-%m" else "%Y-%m")
            # If fmt == '%Y-%m', it returned 1st day by default; ensure day=1:
            if fmt == "%Y-%m":
                dt = dt.replace(day=1)
            return dt.date()
        except ValueError:
            continue

    # Final fallback: lenient pandas parser
    try:
        ts = pd.to_datetime(s, errors="raise")
        # If no day provided, pandas sets the first day of the month by default
        return ts.date()
    except Exception as e:
        raise ValueError(f"unrecognized date format: {cell!r} ({e})")


def _parse_value(cell: str):
    """
    Parse numeric value as Decimal. Returns Decimal or raises.
    """
    s = (str(cell) if cell is not None else "").strip()
    if s == "":
        raise ValueError("empty value")
    # Remove any stray commas
    s = s.replace(",", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        # Try float round-trip as a last resort
        try:
            return Decimal(str(float(s)))
        except Exception:
            raise


class Command(BaseCommand):
    help = "Ingest historical Shiller PE CSV with columns: Date, Value (e.g., Date='1881.01')."

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            required=True,
            help="Path to CSV file containing 'Date' and 'Value' columns.",
        )
        parser.add_argument(
            "--update-existing",
            action="store_true",
            help="If set, update value for rows that already exist.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Parse and report counts, but do not write to the database.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1000,
            help="Bulk create batch size (default: 1000).",
        )

    def handle(self, *args, **opts):
        path = Path(opts["path"]).expanduser()
        update_existing = bool(opts["update_existing"])
        dry_run = bool(opts["dry_run"])
        batch_size = int(opts["batch_size"])

        if not path.exists():
            raise CommandError(f"CSV not found: {path}")

        # Prefer pandas if user has it; fallback to csv module (but we imported pandas anyway)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise CommandError(f"Failed to read CSV via pandas: {e}")

        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        if "Date" not in df.columns or "Value" not in df.columns:
            raise CommandError("CSV must have 'Date' and 'Value' columns")

        # Drop fully empty rows
        df = df.dropna(how="all")

        # Parse columns
        parsed = []
        errors = 0
        for i, row in df.iterrows():
            try:
                dt = _parse_date(str(row["Date"]))
                val = _parse_value(row["Value"])
                parsed.append((dt, val))
            except Exception as e:
                errors += 1
                self.stderr.write(f"Row {i}: skipped due to parse error: {e}")

        if not parsed:
            raise CommandError("No valid rows parsed from CSV.")

        # Deduplicate inside the CSV itself (keep last occurrence)
        dedup_map = {}
        for dt, val in parsed:
            dedup_map[dt] = val
        rows = sorted(dedup_map.items(), key=lambda x: x[0])  # sort by date asc

        to_create = []
        to_update = []
        existing_map = {
            r["date"]: r["value"]
            for r in ShillerPeDaily.objects.filter(date__in=[d for d, _ in rows]).values(
                "date", "value"
            )
        }

        for dt, val in rows:
            if dt in existing_map:
                if update_existing and existing_map[dt] != val:
                    to_update.append((dt, val))
            else:
                to_create.append(ShillerPeDaily(date=dt, value=val))

        created = 0
        updated = 0

        if dry_run:
            self.stdout.write("=== DRY RUN ===")
            self.stdout.write(f"Parsed OK: {len(rows)} rows")
            self.stdout.write(f"Existing:  {len(existing_map)}")
            self.stdout.write(f"To create: {len(to_create)}")
            self.stdout.write(f"To update: {len(to_update)}")
            self.stdout.write(f"Parse errors: {errors}")
            return

        with transaction.atomic():
            # Bulk create in batches
            if to_create:
                for i in range(0, len(to_create), batch_size):
                    chunk = to_create[i : i + batch_size]
                    ShillerPeDaily.objects.bulk_create(chunk, ignore_conflicts=True)
                    created += len(chunk)

            # Update existing if requested
            if to_update:
                # Update one by one (safe and simple)
                for dt, val in to_update:
                    ShillerPeDaily.objects.filter(date=dt).update(value=val)
                updated = len(to_update)

        self.stdout.write(
            self.style.SUCCESS(
                f"Ingest finished — created: {created}, updated: {updated}, parse_errors: {errors}"
            )
        )

        # Print tail
        tail = list(
            ShillerPeDaily.objects.order_by("-date")
            .values("date", "value")[:5]
        )[::-1]
        if tail:
            self.stdout.write("Tail:")
            for r in tail:
                self.stdout.write(f"  {r['date']}  {r['value']}")

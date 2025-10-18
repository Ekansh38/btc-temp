# btc_temp/models.py

from django.db import models


class BitcoinTemperature(models.Model):
    """Stores computed Bitcoin Temperature snapshots with metadata."""
    ts = models.DateTimeField(auto_now_add=True)
    temperature = models.DecimalField(max_digits=18, decimal_places=6)
    inputs = models.JSONField(null=True, blank=True)
    calc_version = models.CharField(max_length=32, default="v1")

    class Meta:
        ordering = ["-ts"]
        indexes = [models.Index(fields=["-ts"])]

    def __str__(self):
        return f"{self.ts.isoformat()} — {self.temperature}"


class MvrvZScoreDaily(models.Model):
    """Daily MVRV Z-score series (core metric)."""
    date = models.DateField(unique=True)  # UTC date
    price = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    market_cap = models.DecimalField(max_digits=30, decimal_places=6, null=True)
    realized_cap = models.DecimalField(max_digits=30, decimal_places=6, null=True)
    zscore = models.DecimalField(max_digits=10, decimal_places=6, null=True)

    class Meta:
        ordering = ["date"]
        indexes = [models.Index(fields=["-date"])]

    def __str__(self):
        return f"MVRV {self.date}: {self.zscore}"


class RhodlRatioDaily(models.Model):
    """Daily RHODL ratio series."""
    date = models.DateField(unique=True)
    value = models.DecimalField(max_digits=20, decimal_places=10, null=True)

    class Meta:
        ordering = ["date"]
        indexes = [models.Index(fields=["-date"])]

    def __str__(self):
        return f"RHODL {self.date}: {self.value}"


class ShillerPeDaily(models.Model):
    """Daily Shiller PE ratio series (stock market valuation metric)."""
    date = models.DateField(unique=True)
    value = models.DecimalField(max_digits=20, decimal_places=10, null=True)

    class Meta:
        ordering = ["date"]
        indexes = [models.Index(fields=["-date"])]

    def __str__(self):
        return f"Shiller PE {self.date}: {self.value}"

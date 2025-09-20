from django.db import models


class BitcoinTemperature(models.Model):
    ts = models.DateTimeField(auto_now_add=True)
    temperature = models.DecimalField(max_digits=18, decimal_places=6)
    inputs = models.JSONField(null=True, blank=True)
    calc_version = models.CharField(max_length=32, default="v1")

    class Meta:
        indexes = [models.Index(fields=["-ts"])]

class MvrvZScoreDaily(models.Model):
    date = models.DateField(unique=True)                 # UTC date
    price = models.DecimalField(max_digits=20, decimal_places=8, null=True)
    market_cap = models.DecimalField(max_digits=30, decimal_places=6, null=True)
    realized_cap = models.DecimalField(max_digits=30, decimal_places=6, null=True)
    zscore = models.DecimalField(max_digits=10, decimal_places=6)

    class Meta:
        ordering = ["date"]
        indexes = [models.Index(fields=["-date"])]


class RhodlRatioDaily(models.Model):
    date = models.DateField(unique=True)
    value = models.DecimalField(max_digits=20, decimal_places=10, null=True)  # core series value
    class Meta:
        ordering = ["date"]
        indexes = [models.Index(fields=["-date"])]

class FundingRatesDaily(models.Model):
    date = models.DateField(unique=True)
    value = models.DecimalField(max_digits=20, decimal_places=10, null=True)  # core series value
    class Meta:
        ordering = ["date"]
        indexes = [models.Index(fields=["-date"])]

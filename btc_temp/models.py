from django.db import models


class BitcoinTemperature(models.Model):
    ts = models.DateTimeField(auto_now_add=True)
    temperature = models.DecimalField(max_digits=18, decimal_places=6)
    inputs = models.JSONField(null=True, blank=True)
    calc_version = models.CharField(max_length=32, default="v1")

    class Meta:
        indexes = [models.Index(fields=["-ts"])]

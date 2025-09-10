from django.contrib import admin
from .models import BitcoinTemperature

@admin.register(BitcoinTemperature)
class BitcoinTemperatureAdmin(admin.ModelAdmin):
    list_display = ("ts", "temperature", "calc_version")
    list_filter = ("calc_version", "ts")
    search_fields = ("calc_version",)
    ordering = ("-ts",)

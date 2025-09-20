from django.contrib import admin
from .models import BitcoinTemperature, MvrvZScoreDaily, RhodlRatioDaily, FundingRatesDaily

@admin.register(BitcoinTemperature)
class BitcoinTemperatureAdmin(admin.ModelAdmin):
    list_display = ("ts", "temperature", "calc_version")
    list_filter = ("calc_version", "ts")


admin.site.register(MvrvZScoreDaily)


@admin.register(RhodlRatioDaily)
class RhodlRatioDailyAdmin(admin.ModelAdmin):
    list_display = ("date", "value")
    ordering = ("-date",)

@admin.register(FundingRatesDaily)
class FundingRatesDailyAdmin(admin.ModelAdmin):
    list_display = ("date", "value")
    ordering = ("-date",)
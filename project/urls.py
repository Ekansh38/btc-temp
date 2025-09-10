from django.urls import path
from btc_temp import views as btc
from django.contrib import admin

urlpatterns = [
    path("admin/", admin.site.urls),

    path("api/bitcoin-temperature/latest", btc.latest),
    path("api/bitcoin-temperature/history", btc.history),
]

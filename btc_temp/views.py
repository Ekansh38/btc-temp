from django.http import JsonResponse
from django.utils.timezone import now
from datetime import timedelta
from .models import BitcoinTemperature


def latest(request):
    m = BitcoinTemperature.objects.order_by("-ts").first()
    if not m:
        return JsonResponse({"ts": None, "temperature": None})
    return JsonResponse({"ts": m.ts.isoformat(), "temperature": float(m.temperature)})


def history(request):
    rng = request.GET.get("range", "30d")
    days = int(rng[:-1]) if rng.endswith("d") else 30
    start = now() - timedelta(days=days)
    rows = (
        BitcoinTemperature.objects.filter(ts__gte=start)
        .order_by("ts")
        .values_list("ts", "temperature")
    )
    return JsonResponse(
        [{"ts": ts.isoformat(), "temperature": float(v)} for ts, v in rows], safe=False
    )


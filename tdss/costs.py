"""Расчёт ожидаемой стоимости поездки на основе прогноза цен (П6)."""
from __future__ import annotations

from .data import TripRequest, load_destinations
from .forecast import price_factor

HOTEL_CLASSES = {  # множитель к суточным расходам, бонус комфорта (баллы полезности)
    "эконом": {"mult": 0.75, "comfort": 0.0},
    "стандарт": {"mult": 1.00, "comfort": 6.0},
    "комфорт": {"mult": 1.50, "comfort": 14.0},
}
ACTIVITY_RESERVE = 0.10   # резерв на экскурсии/входные билеты, доля от бюджета (для отбора)


def trip_cost(dest_id: str, trip: TripRequest, hotel_mult: float = 1.0) -> dict:
    d = load_destinations().set_index("id").loc[dest_id]
    f, f_lo, f_hi = price_factor(dest_id, trip.year, trip.month)
    flight = d.flight_rub * f
    stay = trip.days * d.daily_rub * f * hotel_mult
    visa = float(d.visa_cost_rub)
    total = flight + stay + visa
    return {
        "flight": round(flight, -1), "stay": round(stay, -1), "visa": visa,
        "total": round(total, -1),
        "total_lo": round(d.flight_rub * f_lo + trip.days * d.daily_rub * f_lo * hotel_mult + visa, -1),
        "total_hi": round(d.flight_rub * f_hi + trip.days * d.daily_rub * f_hi * hotel_mult + visa, -1),
        "factor": f, "daily": d.daily_rub * f,
    }

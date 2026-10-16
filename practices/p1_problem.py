"""П1. Постановка задачи: формальное описание и исходные данные."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tdss.ahp import CRITERIA  # noqa: E402
from tdss.data import REPORTS, TripRequest, ensure_dirs, load_destinations, load_pois, load_price_history, load_ratings  # noqa: E402

PROBLEM = {
    "domain": "Туризм: планирование индивидуальной поездки",
    "problem": ("Турист выбирает направление и маршрут, сопоставляя конкурирующие факторы (стоимость, "
                "климат, интересы, доступность, безопасность) при ограниченных бюджете и времени и "
                "неопределённости цен, погоды и срыва поездки."),
    "dm": "Индивидуальный турист из Москвы (ЛПР), самостоятельно планирующий поездку",
    "goal": ("Выбрать направление, стратегию бронирования и маршрут по дням, максимизирующие "
             "ценность поездки для ЛПР при соблюдении ограничений; получить несколько обоснованных вариантов"),
    "constraints": [
        "Бюджет на человека B (перелёт + проживание + питание + экскурсии + трансферы + комиссии)",
        "Длительность поездки D дней и темп (часов активностей в полный день)",
        "Сроки оформления визы и риск отказа (для шенгенских стран)",
        "Сезонность: пляжные объекты допустимы только в купальный сезон",
        "Не более 2 музеев в день; каждый объект посещается не более одного раза",
    ],
    "data_sources": [
        "data/destinations.csv - 10 направлений: базовые цены (экспертная оценка 2026 г.), время перелёта, визы, "
        "приём российских карт, индекс безопасности, профиль направления (7 признаков 0…1)",
        "data/climate.csv - климатические нормы по месяцам (средняя температура, число дождливых дней)",
        "data/pois.csv - 100 объектов показа: координаты, категория, длительность, стоимость билета, рейтинг",
        "data/price_history.csv - помесячная стоимость 7-дневной поездки 2019-01…2026-09 (синтетический ряд "
        "с трендом, курсом, сезонностью и шоками)",
        "data/users.csv, data/ratings.csv - 600 пользователей и их оценки направлений (синтетические)",
    ],
}


def main():
    ensure_dirs()
    trip = TripRequest()
    dest = load_destinations()
    alts = dest[["id", "name", "country", "flight_rub", "daily_rub", "flight_hours", "visa_required",
                 "ru_cards", "safety_index"]].to_dict("records")
    out = {
        **PROBLEM,
        "base_scenario": {
            "month": trip.month, "year": trip.year, "days": trip.days, "budget_per_person": trip.budget_per_person,
            "pace_hours": trip.pace_hours, "profile": trip.profile, "budget_sensitivity": trip.budget_sensitivity,
            "days_until_departure": trip.days_until_departure,
        },
        "alternatives": alts,
        "criteria": [{"id": c[0], "name": c[1], "sense": c[2]} for c in CRITERIA],
        "data_volume": {
            "destinations": len(dest), "pois": len(load_pois()), "price_points": int(load_price_history().cost_7d_rub.notna().sum()),
            "ratings": len(load_ratings()),
        },
        "formal": (
            "A = {a1,…,a10}; K = {K1,…,K6}; веса w (АНР). Для поездки (m, D, B, профиль p):\n"
            "  1) A' = {a ∈ A : БЗ(a, m, D, B) = «допустимо»}  - фильтрация базой знаний;\n"
            "  2) F(a) = 0.7·S_АНР(a)/max S_АНР + 0.3·R(a, p)/max R  - интегральная оценка;\n"
            "  3) s*(a) = argmax_s E[V(a) − C(a, s, ξ)]  - стратегия бронирования при неопределённости ξ;\n"
            "  4) маршрут: max Σ u_i y_it + комфорт − α·время − λ·затраты  при  Σ t_it ≤ H_t,  C ≤ B  (MILP);\n"
            "  5) выдать варианты: «Оптимальный» (max F), «Экономный» (min C при F ≥ 0.85 max F),"
            " «Надёжный» (max критерий Гурвица)."),
    }
    (REPORTS / "p1_problem.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps({k: out[k] for k in ("dm", "goal", "base_scenario", "data_volume")}, ensure_ascii=False, indent=2))
    print(out["formal"])


if __name__ == "__main__":
    main()

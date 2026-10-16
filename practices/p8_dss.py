"""П8. Итоговая СППР: сквозной прогон для трёх сценариев ЛПР и итоговые рекомендации."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tdss import ahp, bi_export, pipeline  # noqa: E402
from tdss.data import REPORTS, TripRequest, ensure_dirs  # noqa: E402

# Важность критериев по шкале 1…9 (быстрый режим АНР: отношения оценок → матрица Саати)
FAMILY_IMPORTANCE = {"K1": 6, "K2": 7, "K3": 2, "K4": 9, "K5": 5, "K6": 8}
COMFORT_IMPORTANCE = {"K1": 3, "K2": 7, "K3": 9, "K4": 2, "K5": 4, "K6": 5}

SCENARIOS = {
    "A. Базовый: май, 5 дней, 70 тыс. ₽, культура и гастрономия": (TripRequest(), None),
    "B. Тот же ЛПР, бюджет 120 тыс. ₽": (TripRequest(budget_per_person=120_000), None),
    "C. Тот же ЛПР: октябрь, 7 дней, 180 тыс. ₽, стоимость второстепенна": (
        TripRequest(month=10, days=7, budget_per_person=180_000, budget_sensitivity=0.2),
        ahp.matrix_from_importance(COMFORT_IMPORTANCE)),
    "D. Семья с детьми: июль, 7 дней, 120 тыс. ₽, пляж": (TripRequest(
        month=7, days=7, budget_per_person=120_000, with_children=True, pace_hours=6, budget_sensitivity=0.5,
        profile={"culture": 0.2, "nature": 0.5, "beach": 0.95, "gastronomy": 0.4, "nightlife": 0.1,
                 "shopping": 0.3, "family": 0.9}), ahp.matrix_from_importance(FAMILY_IMPORTANCE)),
}


def variant_record(v) -> dict:
    t = v.route.totals
    return {"variant": v.kind, "destination": v.name, "score": round(v.score, 3), "strategy": v.strategy,
            "hotel": v.route.hotel_class, "n_pois": t["n_pois"], "utility": t["utility"], "total": t["total"],
            "total_with_fees": v.risk.get("route_total_with_fees"),
            "p_overrun": round(v.risk.get("P(перерасход маршрута)", 0), 4),
            "rationale": v.rationale, "warnings": v.warnings,
            "days": {int(d): g["name"].tolist() for d, g in v.route.plan.groupby("day")}}


def main():
    ensure_dirs()
    out = {}
    for name, (trip, matrix) in SCENARIOS.items():
        t0 = time.time()
        res = pipeline.run(trip, criteria_matrix=matrix, route_time_limit=60)
        out[name] = {
            "params": trip.label(), "elapsed_s": round(time.time() - t0, 1),
            "feasible": res.table[~res.table.excluded].name.tolist(),
            "excluded": res.table[res.table.excluded][["name", "reasons"]].to_dict("records"),
            "ranking": res.table[["rank", "name", "ahp", "rec", "integral", "cost", "status"]].round(4).to_dict("records"),
            "tree_path": res.tree_path, "tree_ev": round(res.tree_ev),
            "criteria_weights": dict(zip(ahp.CRIT_IDS, res.ahp.criteria.weights.round(3).tolist())),
            "criteria_cr": round(res.ahp.criteria.cr, 4),
            "variants": [variant_record(v) for v in res.variants],
        }
        if name.startswith("A."):
            rank = res.table[["rank", "dest_id", "name", "ahp", "rec", "integral", "cost", "status", "reasons"]].round(4)
            plans = pd.concat([v.route.plan.assign(variant=v.kind, destination=v.name) for v in res.variants])
            bi_export.build(rank, plans[["variant", "destination", "day", "order", "name", "category", "arrival",
                                         "duration_h", "ticket_rub", "lat", "lon"]])
        print(f"\n######## {name} ({out[name]['elapsed_s']} с)")
        print("Допустимы:", ", ".join(out[name]["feasible"]))
        for v in out[name]["variants"]:
            print(f"  {v['variant']:<12} {v['destination']:<16} оценка {v['score']:.3f} | {v['hotel']:<8} | "
                  f"{v['total']:,.0f} ₽ | объектов {v['n_pois']} | {v['strategy']} | P(перерасх.) {v['p_overrun']:.1%}"
                  .replace(",", " "))
    (REPORTS / "p8_scenarios.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

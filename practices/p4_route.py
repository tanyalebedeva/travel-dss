"""П4. Оптимизационная модель: маршрут по дням и класс проживания (PuLP, MILP)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tdss import pipeline, route_opt as ro, viz  # noqa: E402
from tdss.data import REPORTS, TripRequest, ensure_dirs, load_destinations  # noqa: E402

plt = viz.plt


def main():
    ensure_dirs()
    trip = TripRequest()
    res = pipeline.run(trip, route_time_limit=60)
    dest = load_destinations().set_index("id")
    rows, plans = [], []
    for v in res.variants:
        r = v.route
        g = ro.greedy_route(v.dest_id, v.pois, trip.days, trip.pace_hours)
        rows.append({"variant": v.kind, "destination": v.name, "status": r.status, "hotel": r.hotel_class,
                     "objective": round(r.objective, 2), **{k: r.totals[k] for k in
                     ("n_pois", "utility", "travel_h", "flight_visa", "stay", "tickets", "transfers", "total", "budget_slack")},
                     "greedy_utility": g["utility"], "greedy_n_pois": g["n_pois"], "greedy_travel_h": g["travel_h"],
                     "gain_vs_greedy_%": round((r.totals["utility"] / g["utility"] - 1) * 100, 1),
                     "solve_s": round(r.solve_time, 2), "candidates": len(v.pois),
                     "hours_used": r.totals["hours_used"], "hours_available": r.totals["hours_available"]})
        plans.append(r.plan.assign(variant=v.kind, destination=v.name))
    summary = pd.DataFrame(rows)
    plan = pd.concat(plans)
    summary.to_csv(REPORTS / "p4_routes_summary.csv", index=False)
    plan.to_csv(REPORTS / "p4_route_plans.csv", index=False)

    # чувствительность: темп и бюджет для варианта «Оптимальный»
    v = res.variants[0]
    sens = []
    for pace in (6, 8, 10):
        for budget in (55_000, 70_000, 90_000):
            r = ro.optimize_route(v.dest_id, v.pois, trip.days, budget * 0.95, v.cost["flight"] + v.cost["visa"],
                                  v.cost["daily"], cost_weight=0.8 * trip.budget_sensitivity, pace=pace, time_limit=60)
            sens.append({"pace_h": pace, "budget": budget, "status": r.status, "hotel": r.hotel_class,
                         "n_pois": r.totals.get("n_pois"), "utility": r.totals.get("utility"),
                         "total": r.totals.get("total")})
    sens = pd.DataFrame(sens)
    sens.to_csv(REPORTS / "p4_sensitivity.csv", index=False)
    (REPORTS / "p4_summary.json").write_text(json.dumps({
        "routes": summary.to_dict("records"), "sensitivity": sens.to_dict("records")},
        ensure_ascii=False, indent=2, default=float))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, v in zip(axes, res.variants):
        p = v.route.plan
        h = dest.loc[v.dest_id]
        allp = v.pois
        ax.scatter(allp.lon, allp.lat, s=18, color=viz.AXIS, zorder=2, label="не вошли")
        for k, (day, d) in enumerate(p.groupby("day")):
            xs = [h.lon] + d.lon.tolist() + [h.lon]
            ys = [h.lat] + d.lat.tolist() + [h.lat]
            ax.plot(xs, ys, color=viz.SERIES[k], lw=1.6, zorder=3)
            ax.scatter(d.lon, d.lat, s=40, color=viz.SERIES[k], edgecolor=viz.SURFACE, linewidth=1.5, zorder=4,
                       label=f"день {day}")
        ax.scatter([h.lon], [h.lat], marker="s", s=60, color=viz.INK, zorder=5, label="отель")
        ax.set_title(f"{v.kind}: {v.name}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="best")
    viz.save(fig, REPORTS / "p4_routes.png")

    pd.set_option("display.width", 250)
    print(summary.drop(columns=["hours_used", "hours_available"]).to_string(index=False))
    for v in res.variants:
        print(f"\n== {v.kind}: {v.name} ({v.route.hotel_class})")
        print(v.route.plan[["day", "order", "name", "arrival", "duration_h", "travel_h", "ticket_rub", "transfer_rub"]].to_string(index=False))
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()

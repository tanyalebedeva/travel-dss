"""П7. Рекомендательный модуль: качество на кросс-валидации, рекомендации и объяснения."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tdss import recommender as rec, viz  # noqa: E402
from tdss.data import REPORTS, TripRequest, ensure_dirs  # noqa: E402

plt = viz.plt


def main():
    ensure_dirs()
    trip = TripRequest()
    ev = rec.evaluate()
    ev["table"].round(4).to_csv(REPORTS / "p7_metrics.csv", index=False)
    ev["grid"].round(4).to_csv(REPORTS / "p7_weight_grid.csv", index=False)
    r = rec.recommend_destinations(trip, ev["best_weights"])
    r.table.to_csv(REPORTS / "p7_destinations.csv", index=False)
    pois = rec.recommend_pois("spb", trip)
    pois[["poi_id", "name", "category", "rating", "fit", "utility", "why"]].to_csv(REPORTS / "p7_pois_spb.csv", index=False)

    # профиль «пляжный семейный» - показать, что рекомендации меняются
    beach = TripRequest(month=7, profile={"culture": 0.2, "nature": 0.5, "beach": 0.95, "gastronomy": 0.4,
                                          "nightlife": 0.2, "shopping": 0.3, "family": 0.9})
    r_beach = rec.recommend_destinations(beach, ev["best_weights"])
    r_beach.table.to_csv(REPORTS / "p7_destinations_beach_july.csv", index=False)

    out = {"best_weights": ev["best_weights"], "cf_rmse": round(ev["cf_rmse"], 3),
           "baseline_rmse": round(ev["baseline_rmse"], 3), "n_test_users": ev["n_test_users"],
           "metrics": ev["table"].round(4).to_dict("records"),
           "top_base": r.table.head(5)[["name", "score", "explanation"]].to_dict("records"),
           "top_beach_july": r_beach.table.head(5)[["name", "score", "explanation"]].to_dict("records")}
    (REPORTS / "p7_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    t = ev["table"].iloc[::-1]
    ax.barh(t.model, t["NDCG@3"], color=[viz.SERIES[0] if m.startswith("Гибрид") else viz.AXIS for m in t.model],
            height=0.55)
    for y, v in enumerate(t["NDCG@3"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8.5, color=viz.INK2)
    ax.set_xlim(0.6, 1.02)
    ax.set_title("NDCG@3 на 5-кратной кросс-валидации по пользователям")
    ax.grid(axis="y", visible=False)
    viz.save(fig, REPORTS / "p7_metrics.png")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 130)
    print(ev["table"].round(4).to_string(index=False))
    print("веса гибрида:", ev["best_weights"], "RMSE CF:", round(ev["cf_rmse"], 3), "RMSE по среднему:", round(ev["baseline_rmse"], 3))
    print(r.table[["rank", "name", "content", "cf_pred", "popularity", "score", "explanation"]].to_string(index=False))
    print(r_beach.table[["rank", "name", "score"]].head(5).to_string(index=False))
    print(pois[["name", "category", "utility", "why"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""П6. Предиктивная аналитика: прогноз стоимости поездки, метрики, витрины для Superset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tdss import bi_export, forecast, viz  # noqa: E402
from tdss.data import REPORTS, dest_name, ensure_dirs  # noqa: E402

plt = viz.plt


def main():
    ensure_dirs()
    r = forecast.run(save=True)
    best, summary, preds, fc = r["best"], r["summary"], r["preds"], r["forecast"]
    wide = forecast.series_matrix()

    # 1) валидация: факт vs прогноз лучшей модели для трёх типичных профилей сезонности
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for ax, d in zip(axes, ["spb", "antalya", "dubai"]):
        s = wide[d].loc["2024-01-01":]
        p = preds[(preds.dest_id == d) & (preds.model == best)]
        ax.plot(s.index, s / 1000, color=viz.INK2, lw=1.6, label="факт")
        ax.fill_between(p.month, p.lo80 / 1000, p.hi80 / 1000, color=viz.SEQ[1], alpha=0.6, lw=0, label="80% ПИ")
        ax.plot(p.month, p.pred / 1000, color=viz.SERIES[0], label="прогноз")
        mape = r["metrics"].query("dest_id == @d and model == @best").MAPE.iloc[0]
        ax.set_title(f"{dest_name(d)}: MAPE {mape:.1f}%", fontsize=10)
        ax.set_ylabel("тыс. ₽")
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].legend(loc="upper left")
    viz.save(fig, REPORTS / "p6_validation.png")

    # 2) история и прогноз по всем направлениям (малые множители)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5), sharex=True)
    for ax, d in zip(axes.flat, wide.columns):
        s = wide[d]
        f = fc[fc.dest_id == d]
        ax.plot(s.index, s / 1000, color=viz.INK2, lw=1.2)
        ax.fill_between(f.month, f.lo80 / 1000, f.hi80 / 1000, color=viz.SEQ[1], alpha=0.6, lw=0)
        ax.plot(f.month, f.forecast / 1000, color=viz.SERIES[0], lw=1.8)
        may = f[f.month == "2027-05-01"].forecast.iloc[0] / 1000
        ax.scatter([pd.Timestamp("2027-05-01")], [may], s=22, color=viz.SERIES[1], zorder=4)
        ax.set_title(f"{dest_name(d)} · май-27: {may:.0f}", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Стоимость 7-дневной поездки, тыс. ₽: история (серый) и прогноз ETS с 80%-м интервалом (синий)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=viz.INK)
    viz.save(fig, REPORTS / "p6_forecast_grid.png")

    # 3) сравнение моделей
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    sm = summary.sort_values("MAPE", ascending=False)
    ax.barh(sm.index, sm.MAPE, color=[viz.SERIES[0] if m == best else viz.AXIS for m in sm.index], height=0.55)
    for y, v in enumerate(sm.MAPE):
        ax.text(v + 0.1, y, f"{v:.2f}%", va="center", fontsize=8.5, color=viz.INK2)
    ax.set_title("Средняя MAPE на отложенных 12 месяцах")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, sm.MAPE.max() * 1.2)
    viz.save(fig, REPORTS / "p6_models.png")

    tables = bi_export.build()
    may = fc[fc.month == "2027-05-01"][["dest_id", "forecast", "lo80", "hi80"]]
    out = {"best_model": best, "summary": summary.round(2).reset_index().to_dict("records"),
           "train": "2021-01…2025-09", "test": "2025-10…2026-09", "forecast_horizon": "2026-10…2027-12",
           "may_2027": may.assign(name=may.dest_id.map(dest_name)).to_dict("records"), "bi_tables": tables}
    (REPORTS / "p6_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(summary)
    print("Лучшая модель:", best)
    print(r["metrics"].pivot(index="dest_id", columns="model", values="MAPE").round(2))
    print(may.to_string(index=False))
    print("Витрины Superset:", tables)


if __name__ == "__main__":
    main()

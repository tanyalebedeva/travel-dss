"""П2. Многокритериальный анализ (АНР): веса, согласованность, оценки, чувствительность."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tdss import ahp, viz  # noqa: E402
from tdss.data import REPORTS, TripRequest, ensure_dirs  # noqa: E402

plt = viz.plt


def fmt_matrix(a: np.ndarray, labels) -> pd.DataFrame:
    def f(x):
        if x >= 1:
            return f"{x:.0f}"
        return f"1/{round(1 / x):.0f}"
    return pd.DataFrame([[f(v) for v in row] for row in a], index=labels, columns=labels)


def main():
    ensure_dirs()
    trip = TripRequest()
    res = ahp.run(trip)
    cp = res.criteria
    crit = pd.DataFrame({"criterion": ahp.CRIT_IDS, "name": [ahp.CRIT_NAMES[c] for c in ahp.CRIT_IDS],
                         "weight_eigen": cp.weights.round(4), "weight_geomean": cp.gm_weights.round(4)})
    fmt_matrix(res.criteria_matrix, ahp.CRIT_IDS).to_csv(REPORTS / "p2_criteria_matrix.csv")
    crit.to_csv(REPORTS / "p2_criteria_weights.csv", index=False)
    res.indicators.to_csv(REPORTS / "p2_indicators.csv", index=False)
    res.local.round(4).to_csv(REPORTS / "p2_local_priorities.csv")
    res.scores.round(4).to_csv(REPORTS / "p2_scores.csv", index=False)
    names = dict(zip(res.indicators.dest_id, res.indicators.name))
    fmt_matrix(res.alt_matrices["K1"], [names[d] for d in res.local.index]).to_csv(REPORTS / "p2_alt_matrix_K1.csv")
    sens = ahp.sensitivity(res)
    switches = ahp.leader_switches(sens, res)
    switches.to_csv(REPORTS / "p2_leader_switches.csv", index=False)
    summary = {
        "lambda_max": round(cp.lambda_max, 4), "CI": round(cp.ci, 4), "CR": round(cp.cr, 4),
        "RI": ahp.RI[6], "weights": dict(zip(ahp.CRIT_IDS, cp.weights.round(4).tolist())),
        "local_CR": {k: round(v, 4) for k, v in res.local_cr.items()},
        "ranking": res.scores[["rank", "name", "score"]].round(4).to_dict("records"),
        "leader_switches": switches.to_dict("records"),
    }
    (REPORTS / "p2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # --- графики ---
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    order = np.argsort(cp.weights)
    ax.barh([f"{ahp.CRIT_IDS[i]} {ahp.CRIT_NAMES[ahp.CRIT_IDS[i]]}" for i in order], cp.weights[order],
            color=viz.SERIES[0], height=0.6)
    for y, v in enumerate(cp.weights[order]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", color=viz.INK2, fontsize=8.5)
    ax.set_title(f"Веса критериев (ОС = {cp.cr:.3f})")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, cp.weights.max() * 1.18)
    viz.save(fig, REPORTS / "p2_weights.png")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sc = res.scores.iloc[::-1]
    ax.barh(sc.name, sc.score, color=viz.SERIES[0], height=0.6)
    for y, v in enumerate(sc.score):
        ax.text(v + 0.001, y, f"{v:.3f}", va="center", color=viz.INK2, fontsize=8.5)
    ax.set_title("Глобальные приоритеты альтернатив (АНР)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, sc.score.max() * 1.15)
    viz.save(fig, REPORTS / "p2_scores.png")

    top = res.scores.dest_id.head(5).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, cid in zip(axes, ["K1", "K3"]):
        g = sens[sens.criterion == cid]
        for k, did in enumerate(top):
            s = g[g.dest_id == did]
            ax.plot(s.weight, s.score, color=viz.SERIES[k], label=names[did])
        ax.axvline(cp.weights[ahp.CRIT_IDS.index(cid)], color=viz.MUTED, lw=1, ls="--")
        ax.set_title(f"Чувствительность к весу {cid}")
        ax.set_xlabel(f"вес {cid} «{ahp.CRIT_NAMES[cid]}»")
    axes[0].set_ylabel("глобальный приоритет")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    viz.save(fig, REPORTS / "p2_sensitivity.png")

    print(fmt_matrix(res.criteria_matrix, ahp.CRIT_IDS))
    print(crit.to_string(index=False))
    print({k: summary[k] for k in ("lambda_max", "CI", "CR")}, "локальные ОС:", summary["local_CR"])
    print(res.scores.round(4).to_string(index=False))
    print(switches.to_string(index=False))


if __name__ == "__main__":
    main()

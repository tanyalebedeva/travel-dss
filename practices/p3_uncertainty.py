"""П3. Риски и неопределённость: сценарии, дерево решений, критерии, пороги, Монте-Карло."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tdss import pipeline, uncertainty as un, viz  # noqa: E402
from tdss.data import REPORTS, TripRequest, ensure_dirs  # noqa: E402

plt = viz.plt
MARK = {"decision": "s", "chance": "o", "terminal": "^"}


def draw_tree(g, nodes_by_level, path_nodes, fname, title, label_fn, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    pos = {}
    for lvl, nodes in enumerate(nodes_by_level):
        n = len(nodes)
        for k, nd in enumerate(nodes):
            pos[nd] = (lvl, 1 - (k + 0.5) / n)
    for u, v in g.edges():
        if u in pos and v in pos:
            on = u in path_nodes and v in path_nodes
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=viz.SERIES[0] if on else viz.AXIS, lw=2.2 if on else 1.0, zorder=1)
            if "prob" in g.edges[u, v]:
                mx, my = (pos[u][0] * 0.35 + pos[v][0] * 0.65), (pos[u][1] * 0.35 + pos[v][1] * 0.65)
                ax.text(mx, my + 0.012, f"{g.edges[u, v]['prob']:.2f}", fontsize=7, color=viz.MUTED, ha="center")
    last = set(nodes_by_level[-1])
    for nd, (x, y) in pos.items():
        kind = g.nodes[nd]["kind"]
        on = nd in path_nodes
        ax.scatter([x], [y], marker=MARK[kind], s=90, color=viz.SERIES[0] if on else viz.SURFACE,
                   edgecolor=viz.SERIES[0] if on else viz.INK2, linewidth=1.3, zorder=3)
        if kind == "terminal" or nd in last:
            ax.text(x + 0.06, y, label_fn(nd), fontsize=7.6, va="center", color=viz.INK if on else viz.INK2)
        else:
            ax.text(x, y + 0.035, label_fn(nd), fontsize=7.6, va="bottom", ha="center",
                    color=viz.INK if on else viz.INK2,
                    bbox=dict(boxstyle="round,pad=0.15", fc=viz.SURFACE, ec="none", alpha=0.9))
    ax.set_title(title)
    ax.axis("off")
    ax.set_xlim(-0.3, len(nodes_by_level) - 0.1 + 0.9)
    ax.set_ylim(-0.02, 1.06)
    viz.save(fig, REPORTS / fname)


def main():
    ensure_dirs()
    trip = TripRequest()
    res = pipeline.run(trip, with_routes=False)
    alts = [{"dest_id": x.dest_id, "name": x["name"], "score": x.integral, "cost": x.cost,
             "cost_lo": x.cost_lo, "cost_hi": x.cost_hi, "disruption_risk": x.disruption_risk,
             "fees_rate": x.fees_rate}
            for _, x in res.table.iterrows() if f"{x['name']} | " in " ".join(res.payoff.index)]
    inputs = un.build_inputs(alts, trip.budget_per_person, trip.month, max(trip.months_ahead - 1, 1))
    factors = pd.DataFrame([{
        "dest_id": i.dest_id, "name": i.name, "cost_now": round(i.cost, -1), "value": round(i.value, -1),
        "sigma_price": round(i.sigma, 4),
        "late_low": round(un._cost_late(i, -1), -1), "late_mid": round(un._cost_late(i, 0), -1),
        "late_high": round(un._cost_late(i, 1), -1),
        "q_disruption": i.q, "p_bad_weather": round(i.p_bad, 3), "weather_sensitivity": round(i.ws, 3),
    } for i in inputs])
    factors.to_csv(REPORTS / "p3_factors.csv", index=False)
    res.payoff.round(0).to_csv(REPORTS / "p3_payoff_matrix.csv")
    res.prob.round(5).to_csv(REPORTS / "p3_state_probs.csv")
    res.criteria.round(0).to_csv(REPORTS / "p3_criteria_values.csv")
    res.best_by_criteria.round(0).to_csv(REPORTS / "p3_best_by_criteria.csv", index=False)
    res.mc.round(4).to_csv(REPORTS / "p3_monte_carlo.csv", index=False)
    ev_table = pd.DataFrame([{"Направление": i.name, **{un.STRATEGIES[s]: round(un.ev(i, s), 0) for s in un.STRATEGIES}}
                             for i in inputs])
    ev_table.to_csv(REPORTS / "p3_ev_by_strategy.csv", index=False)
    sw = un.hurwicz_switches(res.payoff)
    summary = {
        "tree_ev": round(res.tree_ev, 0), "tree_path": res.tree_path,
        "thresholds": {k: {kk: (None if pd.isna(vv) else round(vv, 4)) for kk, vv in v.items()}
                       for k, v in res.thresholds.items()},
        "hurwicz_switches": sw,
        "best_by_criteria": res.best_by_criteria.round(0).to_dict("records"),
        "assumptions": {"refund_premium": un.REFUND_PREMIUM, "nonref_loss": un.NONREF_LOSS, "ref_loss": un.REF_LOSS,
                        "late_premium": un.LATE_PREMIUM, "late_disrupt_after": un.LATE_DISRUPT_AFTER,
                        "value_factor": un.VALUE_FACTOR, "price_probs": un.PRICE_PROBS.round(4).tolist()},
    }
    (REPORTS / "p3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    # --- деревья ---
    g = un.build_tree(inputs)
    un.rollback(g)
    lvl1 = list(g.successors("root"))
    lvl2 = [s for a in lvl1 for s in g.successors(a)]
    best_a = g.nodes["root"]["best"]
    best_s = g.nodes[best_a]["best"]
    k = lambda v: f"{v/1000:,.1f} тыс.".replace(",", " ")
    draw_tree(g, [["root"], lvl1, lvl2], {"root", best_a, best_s}, "p3_tree_overview.png",
              f"Дерево решений: выбор направления и стратегии (EV = {k(res.tree_ev)} ₽)",
              lambda n: (g.nodes[n]["label"] if g.nodes[n]["kind"] == "decision" and n == "root"
                         else f"{g.nodes[n]['label']}: EV {k(g.nodes[n]['ev'])}"), (9.5, 6.2))
    late = f"{best_a}/S3"
    sub1 = list(g.successors(late))
    sub2 = [c for p in sub1 for c in g.successors(p)]
    sub3 = [c for d in sub2 if g.nodes[d]["kind"] != "terminal" for c in g.successors(d)]
    draw_tree(g, [[late], sub1, sub2, sub3], set(), "p3_tree_detail.png",
              f"Ветвь «{g.nodes[best_a]['label']} - {g.nodes[late]['label']}»: цена, срыв и погода",
              lambda n: (f"{g.nodes[n]['label']}: {k(g.nodes[n]['payoff'])}" if g.nodes[n]["kind"] == "terminal"
                         else f"{g.nodes[n]['label']}: EV {k(g.nodes[n]['ev'])}"), (11, 6.0))

    pd.set_option("display.width", 220)
    print(factors.to_string(index=False))
    print(ev_table.to_string(index=False))
    print(res.best_by_criteria.round(0).to_string(index=False))
    print("Дерево:", res.tree_path, round(res.tree_ev))
    print("Пороги:", summary["thresholds"])
    print("Гурвиц, смена решения при α:", sw)
    print(res.mc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

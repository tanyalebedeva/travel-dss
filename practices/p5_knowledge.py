"""П5. База знаний: онтология OWL + SWRL (Pellet) и продукционные правила для поездки ЛПР."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from tdss import knowledge as kb  # noqa: E402
from tdss.costs import trip_cost  # noqa: E402
from tdss.data import KB_DIR, REPORTS, TripRequest, ensure_dirs, load_destinations  # noqa: E402


def main():
    ensure_dirs()
    onto = kb.build_ontology()
    stats = {
        "classes": len(list(onto.classes())),
        "object_properties": len(list(onto.object_properties())),
        "data_properties": len(list(onto.data_properties())),
        "individuals": len(list(onto.individuals())),
        "swrl_rules": len(kb.SWRL_RULES),
        "production_rules": len(kb.RULES),
    }
    inferred, engine = kb.reason(onto)
    kb.save_inferred(inferred, engine)
    check = kb._python_inference()
    same = inferred.drop(columns="name").astype(str).equals(check.drop(columns="name").astype(str))
    inferred.to_csv(REPORTS / "p5_inferred_facts.csv", index=False)

    trip = TripRequest()
    kb._INFERRED_CACHE = None
    trace, status = [], []
    for d in load_destinations().id:
        inf = kb.evaluate(d, trip, trip_cost(d, trip))
        for rid, name, msg in inf.fired:
            trace.append({"dest_id": d, "rule": rid, "name": name, "message": msg})
        status.append({"dest_id": d, "status": inf.facts["status"], "excluded": inf.excluded,
                       "reasons": "; ".join(inf.facts["reasons"]), "total_with_fees": inf.facts["total_with_fees"],
                       "disruption_risk": inf.facts["disruption_risk"],
                       "exclude_categories": ",".join(inf.facts["exclude_categories"]),
                       "warnings": len(inf.warnings)})
    trace, status = pd.DataFrame(trace), pd.DataFrame(status)
    trace.to_csv(REPORTS / "p5_rule_trace.csv", index=False)
    status.to_csv(REPORTS / "p5_status.csv", index=False)
    rules = pd.DataFrame([{"id": r.rid, "name": r.name, "if": " И ".join(f"{a} {o} {v}" for a, o, v in r.conditions),
                           "then": "; ".join(k for k, _ in r.actions), "priority": r.priority} for r in kb.RULES])
    rules.to_csv(REPORTS / "p5_rules.csv", index=False)
    summary = {"engine": engine, "pellet_matches_python": bool(same), "ontology": stats,
               "owl_file": str((KB_DIR / "travel_dss.owl").relative_to(KB_DIR.parent)),
               "fired_rules_total": len(trace), "excluded": status[status.excluded].dest_id.tolist()}
    (REPORTS / "p5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(inferred.to_string(index=False))
    print(status.to_string(index=False))
    print(trace[trace.dest_id.isin(["rome", "dubai", "spb"])].to_string(index=False))


if __name__ == "__main__":
    main()

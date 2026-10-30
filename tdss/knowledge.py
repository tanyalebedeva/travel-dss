"""П5. База знаний: онтология OWL (Protégé) + продукционные правила с прямым логическим выводом.

1) Онтология travel_dss.owl - классы, отношения, факты (индивиды) и SWRL-правила.
   Логический вывод - ризонер Pellet (через owlready2, нужна Java). Если Java недоступна,
   используется эквивалентный вывод на Python (fallback), результаты совпадают.
2) Продукционная система - правила «ЕСЛИ … ТО …» для конкретной поездки ЛПР
   (бюджет, сроки, сезон), прямой вывод (forward chaining) с журналом срабатываний.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field

import pandas as pd

from .data import KB_DIR, MONTHS_RU, TripRequest, climate_for, ensure_dirs, load_climate, load_destinations, load_pois

IRI = "http://example.org/travel-dss.owl"
CATEGORY_CLASS = {"museum": "Museum", "landmark": "Landmark", "nature": "NatureSite", "beach": "Beach",
                  "gastronomy": "Gastronomy", "entertainment": "Entertainment", "shopping": "Shopping"}

SWRL_RULES = [
    # 1–2. Комфортный месяц: 16…28 °C и не более 9 дождливых дней
    "ClimateRecord(?r), meanTemp(?r, ?t), rainDays(?r, ?n), greaterThanOrEqual(?t, 16.0), "
    "lessThanOrEqual(?t, 28.0), lessThanOrEqual(?n, 9.0) -> ComfortableRecord(?r)",
    "ComfortableRecord(?r), climateOf(?r, ?d), forMonth(?r, ?m) -> comfortableIn(?d, ?m)",
    # 3–4. Купальный сезон пляжного направления: средняя суточная температура от 18 °C
    #      и не более 13 дождливых дней (так в сезон попадают и Чёрное, и Балтийское море)
    "ClimateRecord(?r), meanTemp(?r, ?t), rainDays(?r, ?n), greaterThanOrEqual(?t, 18.0), "
    "lessThanOrEqual(?n, 13.0) -> WarmDryRecord(?r)",
    "WarmDryRecord(?r), climateOf(?r, ?d), BeachDestination(?d), forMonth(?r, ?m) -> beachSeasonIn(?d, ?m)",
    # 5–6. Жаркий месяц: выше 30 °C
    "ClimateRecord(?r), meanTemp(?r, ?t), greaterThan(?t, 30.0) -> HotRecord(?r)",
    "HotRecord(?r), climateOf(?r, ?d), forMonth(?r, ?m) -> hotIn(?d, ?m)",
    # 7. Дальнее направление: перелёт дольше 5 ч
    "Destination(?d), flightHours(?d, ?h), greaterThan(?h, 5.0) -> LongHaulDestination(?d)",
    # 8. Нужны наличные, если в стране не принимают российские карты
    "Destination(?d), locatedIn(?d, ?c), russianCardsAccepted(?c, false) -> CashOnlyDestination(?d)",
]


# =====================================================================================
# 1. Онтология
# =====================================================================================

def build_ontology(path=None):
    """Строит онтологию предметной области и сохраняет её в OWL (RDF/XML) для Protégé."""
    from owlready2 import (DataProperty, FunctionalProperty, Imp, ObjectProperty, Thing,
                           ConstrainedDatatype, get_ontology, AllDifferent)
    ensure_dirs()
    path = path or KB_DIR / "travel_dss.owl"
    onto = get_ontology(IRI + "#")
    dest_df, clim, pois = load_destinations(), load_climate(), load_pois()

    with onto:
        class Country(Thing): pass
        class Destination(Thing): pass
        class Attraction(Thing): pass
        class ClimateRecord(Thing): pass
        class Month(Thing): pass
        class ComfortableRecord(ClimateRecord): pass
        class WarmDryRecord(ClimateRecord): pass
        class HotRecord(ClimateRecord): pass
        for cname in CATEGORY_CLASS.values():
            type(cname, (Attraction,), {})

        class locatedIn(Destination >> Country, FunctionalProperty): pass
        class hasAttraction(Destination >> Attraction): pass
        class attractionOf(Attraction >> Destination):
            inverse_property = hasAttraction
        class climateOf(ClimateRecord >> Destination, FunctionalProperty): pass
        class forMonth(ClimateRecord >> Month, FunctionalProperty): pass

        class isDomestic(Country >> bool, FunctionalProperty): pass
        class requiresVisaForRU(Country >> bool, FunctionalProperty): pass
        class russianCardsAccepted(Country >> bool, FunctionalProperty): pass
        class flightHours(Destination >> float, FunctionalProperty): pass
        class directFlight(Destination >> bool, FunctionalProperty): pass
        class safetyIndex(Destination >> int, FunctionalProperty): pass
        class dailyCost(Destination >> float, FunctionalProperty): pass
        class cultureScore(Destination >> float, FunctionalProperty): pass
        class natureScore(Destination >> float, FunctionalProperty): pass
        class month(ClimateRecord >> int, FunctionalProperty): pass
        class meanTemp(ClimateRecord >> float, FunctionalProperty): pass
        class rainDays(ClimateRecord >> float, FunctionalProperty): pass
        class rating(Attraction >> float, FunctionalProperty): pass
        class durationHours(Attraction >> float, FunctionalProperty): pass
        class ticketCost(Attraction >> float, FunctionalProperty): pass
        # выводимые отношения (SWRL)
        class comfortableIn(Destination >> Month): pass
        class beachSeasonIn(Destination >> Month): pass
        class hotIn(Destination >> Month): pass

        # определяемые классы (вывод классификацией)
        class BeachDestination(Destination):
            equivalent_to = [Destination & hasAttraction.some(onto.Beach)]
        class CultureDestination(Destination):
            equivalent_to = [Destination & cultureScore.some(ConstrainedDatatype(float, min_inclusive=0.7))]
        class NatureDestination(Destination):
            equivalent_to = [Destination & natureScore.some(ConstrainedDatatype(float, min_inclusive=0.7))]
        class DomesticDestination(Destination):
            equivalent_to = [Destination & locatedIn.some(Country & isDomestic.value(True))]
        class ForeignDestination(Destination):
            equivalent_to = [Destination & locatedIn.some(Country & isDomestic.value(False))]
        class VisaDestination(Destination):
            equivalent_to = [Destination & locatedIn.some(Country & requiresVisaForRU.value(True))]
        class FamilyFriendlyDestination(Destination):
            equivalent_to = [Destination & safetyIndex.some(ConstrainedDatatype(int, min_inclusive=65))
                             & hasAttraction.some(onto.Beach | onto.Entertainment)]
        class LongHaulDestination(Destination): pass
        class CashOnlyDestination(Destination): pass

        # ---- факты (индивиды) ----
        countries = {}
        for cname, grp in dest_df.groupby("country"):
            r = grp.iloc[0]
            c = Country(_slug(cname))
            c.label = [cname]
            c.isDomestic = bool(r.is_domestic)
            c.requiresVisaForRU = bool(r.visa_required)
            c.russianCardsAccepted = bool(r.ru_cards)
            countries[cname] = c
        for _, r in dest_df.iterrows():
            d = Destination(r.id)
            d.label = [r["name"]]
            d.locatedIn = countries[r.country]
            d.flightHours = float(r.flight_hours)
            d.directFlight = bool(r.direct_flight)
            d.safetyIndex = int(r.safety_index)
            d.dailyCost = float(r.daily_rub)
            d.cultureScore = float(r.culture)
            d.natureScore = float(r.nature)
        for _, p in pois.iterrows():
            cls = getattr(onto, CATEGORY_CLASS[p.category])
            a = cls(p.poi_id)
            a.label = [p["name"]]
            a.rating = float(p.rating)
            a.durationHours = float(p.duration_h)
            a.ticketCost = float(p.cost_rub)
            onto[p.dest_id].hasAttraction.append(a)
        months = {}
        for m in range(1, 13):
            mi = Month(f"m{m:02d}")
            mi.label = [MONTHS_RU[m - 1]]
            months[m] = mi
        AllDifferent(list(months.values()))
        for _, c in clim.iterrows():
            rec = ClimateRecord(f"clim_{c.dest_id}_{int(c.month):02d}")
            rec.climateOf = onto[c.dest_id]
            rec.forMonth = months[int(c.month)]
            rec.month = int(c.month)
            rec.meanTemp = float(c.temp_c)
            rec.rainDays = float(c.rain_days)
        AllDifferent(list(Destination.instances()))

        for text in SWRL_RULES:
            rule = Imp()
            rule.set_as_rule(text)

    onto.save(file=str(path), format="rdfxml")
    return onto


def _slug(s: str) -> str:
    tr = {"Россия": "Russia", "Турция": "Turkey", "ОАЭ": "UAE", "Грузия": "Georgia", "Италия": "Italy"}
    return tr.get(s, s)


def reason(onto=None) -> tuple[pd.DataFrame, str]:
    """Логический вывод Pellet. Возвращает таблицу выведенных фактов и имя движка вывода."""
    onto = onto or build_ontology()
    engine = "Pellet (owlready2)"
    try:
        from owlready2 import sync_reasoner_pellet
        with onto:
            sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=False, debug=0)
    except Exception as exc:  # нет Java / ошибка ризонера
        engine = f"Python fallback ({type(exc).__name__})"
        return _python_inference(), engine
    rows = []
    defined = ["BeachDestination", "CultureDestination", "NatureDestination", "DomesticDestination",
               "ForeignDestination", "VisaDestination", "FamilyFriendlyDestination", "LongHaulDestination"]
    for d in onto.Destination.instances():
        anc = d.INDIRECT_is_a
        classes = [c for c in defined if getattr(onto, c) in anc]
        rows.append({
            "dest_id": d.name, "name": d.label[0],
            "classes": ", ".join(sorted(set(classes))),
            "comfortable_months": sorted(int(m.name[1:]) for m in set(d.comfortableIn)),
            "beach_season_months": sorted(int(m.name[1:]) for m in set(d.beachSeasonIn)),
            "hot_months": sorted(int(m.name[1:]) for m in set(d.hotIn)),
            "needs_visa": onto.VisaDestination in anc,
            "needs_cash": onto.CashOnlyDestination in anc,
        })
    onto.save(file=str(KB_DIR / "travel_dss_inferred.owl"), format="rdfxml")
    return pd.DataFrame(rows).sort_values("dest_id").reset_index(drop=True), engine


def _python_inference() -> pd.DataFrame:
    """Те же правила, что SWRL/определяемые классы, реализованные на Python."""
    dest, clim, pois = load_destinations(), load_climate(), load_pois()
    rows = []
    for _, d in dest.iterrows():
        c = clim[clim.dest_id == d.id]
        cats = set(pois[pois.dest_id == d.id].category)
        is_beach = "beach" in cats
        classes = ["DomesticDestination" if d.is_domestic else "ForeignDestination"]
        if is_beach: classes.append("BeachDestination")
        if d.culture >= 0.7: classes.append("CultureDestination")
        if d.nature >= 0.7: classes.append("NatureDestination")
        if d.visa_required: classes.append("VisaDestination")
        if d.flight_hours > 5: classes.append("LongHaulDestination")
        if d.safety_index >= 65 and (cats & {"beach", "entertainment"}): classes.append("FamilyFriendlyDestination")
        rows.append({
            "dest_id": d.id, "name": d["name"], "classes": ", ".join(sorted(classes)),
            "comfortable_months": sorted(c[(c.temp_c >= 16) & (c.temp_c <= 28) & (c.rain_days <= 9)].month.astype(int)),
            "beach_season_months": sorted(c[(c.temp_c >= 18) & (c.rain_days <= 13)].month.astype(int)) if is_beach else [],
            "hot_months": sorted(c[c.temp_c > 30].month.astype(int)),
            "needs_visa": bool(d.visa_required), "needs_cash": not bool(d.ru_cards),
        })
    return pd.DataFrame(rows).sort_values("dest_id").reset_index(drop=True)


_INFERRED_CACHE: pd.DataFrame | None = None


INFERRED_PATH = KB_DIR / "inferred_facts.json"


def save_inferred(df: pd.DataFrame, engine: str) -> None:
    ensure_dirs()
    df.assign(engine=engine).to_json(INFERRED_PATH, orient="records", force_ascii=False, indent=1)


def ontology_facts(use_reasoner: bool = False) -> pd.DataFrame:
    """Выведенные онтологические факты: результат Pellet, сохранённый практикой П5
    (kb/inferred_facts.json); если файла нет - эквивалентный вывод на Python."""
    global _INFERRED_CACHE
    if _INFERRED_CACHE is None:
        if use_reasoner:
            _INFERRED_CACHE = reason()[0]
        elif INFERRED_PATH.exists():
            _INFERRED_CACHE = pd.read_json(INFERRED_PATH, orient="records").drop(columns="engine")
        else:
            _INFERRED_CACHE = _python_inference()
    return _INFERRED_CACHE


# =====================================================================================
# 2. Продукционная система
# =====================================================================================

OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le,
       ">": operator.gt, ">=": operator.ge, "in": lambda a, b: a in b, "has": lambda a, b: b in a}


@dataclass
class Rule:
    rid: str
    name: str
    conditions: list            # [(факт, оператор, значение | "@факт")]
    actions: list               # [(факт, значение | callable(facts))]
    message: str = ""
    priority: int = 50

    def matches(self, f: dict) -> bool:
        for key, op, val in self.conditions:
            if key not in f:
                return False
            if isinstance(val, str) and val.startswith("@"):
                val = f.get(val[1:])
                if val is None:
                    return False
            if not OPS[op](f[key], val):
                return False
        return True


RULES: list[Rule] = [
    Rule("R0", "Базовый риск срыва поездки", [("is_domestic", "==", True)],
         [("disruption_risk", 0.03)], priority=99),
    Rule("R0b", "Базовый риск срыва поездки (зарубежье)", [("is_domestic", "==", False)],
         [("disruption_risk", 0.05)], priority=99),
    Rule("R1", "Визу не успеть оформить", [("needs_visa", "==", True), ("days_until", "<", "@visa_days")],
         [("status", "недопустимо"), ("reasons", lambda f: f["reasons"] + ["не успеть оформить визу"])],
         "До поездки меньше срока оформления визы", 90),
    Rule("R2", "Нужна виза", [("needs_visa", "==", True), ("days_until", ">=", "@visa_days")],
         [("visa_refusal_risk", 0.10)],
         "Нужна шенгенская виза: подать документы не позднее чем за {visa_days} дн. до вылета; риск отказа ≈10%", 80),
    Rule("R3", "Российские карты не принимаются", [("needs_cash", "==", True)],
         [("fees_rate", 0.03)],
         "Российские карты не работают: взять наличную валюту или карту иностранного банка (+3% на обмен)", 80),
    Rule("R4", "Стоимость с учётом комиссий",
         [("total", ">", 0)],
         [("total_with_fees", lambda f: round(f["total"] * (1 + f.get("fees_rate", 0.0)), -1))], priority=70),
    Rule("R5", "Превышение бюджета", [("total_with_fees", ">", "@budget")],
         [("status", "недопустимо"), ("reasons", lambda f: f["reasons"] + ["ожидаемая стоимость выше бюджета"])],
         "Ожидаемая стоимость {total_with_fees} ₽ превышает бюджет {budget} ₽", 60),
    Rule("R6", "Бюджет почти исчерпан", [("total_with_fees", ">", "@budget_85"), ("total_with_fees", "<=", "@budget")],
         [("budget_tight", True)], "Запас бюджета меньше 15%: на экскурсии и непредвиденные расходы мало средств", 60),
    Rule("R7", "Жаркий месяц", [("hot_month", "==", True)],
         [("climate_penalty", 0.3), ("outdoor_factor", 0.7)],
         "Жара (≈{temp} °C): прогулки переносить на утро и вечер", 50),
    Rule("R8", "Холодный месяц", [("temp", "<", 8)],
         [("climate_penalty", 0.2), ("exclude_categories", lambda f: sorted(set(f["exclude_categories"]) | {"beach"}))],
         "Холодно (≈{temp} °C): пляжные объекты исключены из маршрута", 50),
    Rule("R9", "Купальный сезон не начался", [("is_beach", "==", True), ("beach_season", "==", False)],
         [("exclude_categories", lambda f: sorted(set(f["exclude_categories"]) | {"beach"}))],
         "Купальный сезон в {month_gen} ещё не начался: пляжи исключены из маршрута", 50),
    Rule("R10", "Купальный сезон", [("is_beach", "==", True), ("beach_season", "==", True)],
         [("beach_bonus", True)], "Купальный сезон: пляжные объекты получают приоритет", 50),
    Rule("R11", "Перелёт со стыковкой", [("direct_flight", "==", False)],
         [("disruption_risk", lambda f: round(f["disruption_risk"] + 0.02, 4))],
         "Нет прямого рейса: перелёт со стыковкой, выше риск задержек", 45),
    Rule("R12", "Нерационально далеко для короткой поездки", [("long_haul", "==", True), ("trip_days", "<=", 4)],
         [("status", "нерационально"), ("reasons", lambda f: f["reasons"] + ["слишком далеко для короткой поездки"])],
         "Перелёт дольше 5 ч при поездке до 4 дней", 45),
    Rule("R13", "Безопасность при поездке с детьми", [("with_children", "==", True), ("safety", "<", 60)],
         [("family_warning", True)], "С детьми: индекс безопасности ниже 60 - выбирать проверенные районы", 40),
    Rule("R14", "Дождливый месяц", [("rain_days", ">=", 12)],
         [("outdoor_factor", 0.8)], "Дождливый месяц ({rain_days} дн.): держать в плане запасные активности в помещениях", 40),
    Rule("R15", "Итоговый риск срыва с учётом визы", [("visa_refusal_risk", ">", 0)],
         [("disruption_risk", lambda f: round(1 - (1 - f["disruption_risk"]) * (1 - f["visa_refusal_risk"]), 4))],
         priority=30),
    Rule("R17", "Жара опасна для детей", [("with_children", "==", True), ("hot_month", "==", True)],
         [("status", "недопустимо"), ("reasons", lambda f: f["reasons"] + ["жара выше 30 °C при поездке с детьми"])],
         "Средняя температура выше 30 °C - направление не рекомендуется для поездки с детьми", 55),
    Rule("R16", "Исключение недопустимой альтернативы", [("status", "==", "недопустимо")],
         [("excluded", True)], priority=10),
]


@dataclass
class Inference:
    dest_id: str
    facts: dict
    fired: list = field(default_factory=list)      # [(rid, name, message)]

    @property
    def warnings(self) -> list[str]:
        return [m for _, _, m in self.fired if m]

    @property
    def excluded(self) -> bool:
        return bool(self.facts.get("excluded"))


def forward_chain(facts: dict, rules: list[Rule] = RULES, max_iter: int = 50) -> tuple[dict, list]:
    """Прямой вывод: на каждом шаге срабатывает правило с наивысшим приоритетом
    из конфликтного набора (каждое правило - не более одного раза)."""
    f = dict(facts)
    fired, used = [], set()
    for _ in range(max_iter):
        agenda = [r for r in rules if r.rid not in used and r.matches(f)]
        if not agenda:
            break
        r = max(agenda, key=lambda x: x.priority)
        for key, val in r.actions:
            f[key] = val(f) if callable(val) else val
        used.add(r.rid)
        msg = r.message.format(**{k: _fmt(v) for k, v in f.items()}) if r.message else ""
        fired.append((r.rid, r.name, msg))
    return f, fired


def _fmt(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", "\u202f")
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"


def initial_facts(dest_id: str, trip: TripRequest, cost: dict) -> dict:
    d = load_destinations().set_index("id").loc[dest_id]
    onto_f = ontology_facts().set_index("dest_id").loc[dest_id]
    t, rain = climate_for(dest_id, trip.month)
    return {
        "dest_id": dest_id, "is_domestic": bool(d.is_domestic), "needs_visa": bool(onto_f.needs_visa),
        "needs_cash": bool(onto_f.needs_cash), "visa_days": int(d.visa_days), "days_until": trip.days_until_departure,
        "total": float(cost["total"]), "budget": float(trip.budget_per_person),
        "budget_85": 0.85 * trip.budget_per_person, "temp": t, "rain_days": rain,
        "hot_month": trip.month in onto_f.hot_months, "is_beach": "BeachDestination" in onto_f.classes,
        "beach_season": trip.month in onto_f.beach_season_months, "direct_flight": bool(d.direct_flight),
        "long_haul": "LongHaulDestination" in onto_f.classes, "trip_days": trip.days,
        "with_children": trip.with_children, "safety": int(d.safety_index),
        "month_gen": ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе",
                      "сентябре", "октябре", "ноябре", "декабре"][trip.month - 1],
        "status": "допустимо", "reasons": [], "exclude_categories": [], "climate_penalty": 0.0,
        "outdoor_factor": 1.0, "fees_rate": 0.0, "visa_refusal_risk": 0.0,
    }


def evaluate(dest_id: str, trip: TripRequest, cost: dict) -> Inference:
    facts, fired = forward_chain(initial_facts(dest_id, trip, cost))
    return Inference(dest_id, facts, fired)


def month_label(m: int) -> str:
    return MONTHS_RU[m - 1]

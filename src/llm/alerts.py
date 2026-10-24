"""
Phase 8 bonus: plain-English delay alerts for customers.

Two modes:
* "llm"      - if the `anthropic` package is installed and ANTHROPIC_API_KEY is
               set, Claude rewrites the prediction into a short, polite message.
* "template" - always available, free, deterministic. Used as the fallback and
               in tests, so the feature never breaks during a demo.

Guardrail: the LLM only REPHRASES facts we give it (ETA, risk, reasons). It is
told not to invent causes or promises, and every number it may use comes from
our model's output.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("ARRIVO_LLM_MODEL", "claude-haiku-5-5")

SYSTEM_PROMPT = (
    "You write short delivery-status messages from a logistics company to its customer. "
    "Use ONLY the facts provided. Do not invent causes, compensation or guarantees. "
    "Tone: {tone}. Length: 2-4 sentences, no bullet points, no emojis. "
    "Mention the expected arrival time and, if the shipment is at risk, the main reason in everyday words."
)


def _fmt_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%a %d %b, %I:%M %p").replace(" 0", " ")
    except ValueError:
        return iso


def _plain_reason(text: str) -> str:
    """'Heavy rain at destination (80 mm) (+6.0 hrs)' -> 'heavy rain at destination'"""
    core = re.sub(r"\s*\([^)]*\)", "", text).strip()     # drop "(80 mm)", "(+6.0 hrs)"
    return core[0].lower() + core[1:] if core else core


# How each reason group should be phrased to a CUSTOMER (ops wording like
# "route's track record" or "supplier reliability" is not customer-friendly).
CUSTOMER_PHRASES = {
    "route_history": "congestion that is common on this route",
    "supplier": "loading time at the dispatch point",
    "festival": "festival-season traffic", "season": "monsoon conditions",
    "dispatch_time": "rush-hour traffic at dispatch", "route_length": "the long distance involved",
    "schedule": "a tight delivery schedule", "terrain": "hilly terrain on the route",
    "cargo": "the handling needs of this cargo", "vehicle": "vehicle restrictions on the route",
    "wind": "strong winds", "temperature": "weather conditions", "weekday": "day-of-week traffic",
    "regions": "conditions on this corridor",
}


def customer_reason(reason: dict) -> str:
    if reason.get("factor") == "rain":
        return _plain_reason(reason["text"])
    return CUSTOMER_PHRASES.get(reason.get("factor"), "conditions on the route")


def template_alert(pred: dict, customer_name: str = "Customer") -> str:
    eta = _fmt_time(pred["expected_arrival"])
    planned = _fmt_time(pred["planned_arrival"])
    sid = pred.get("shipment_id", "your shipment")
    risky = [r for r in pred.get("top_reasons", []) if r.get("risk_direction") == "raises risk"]
    if pred["risk_level"] == "Low":
        return (f"Hi {customer_name}, shipment {sid} from {pred['origin_city']} to {pred['destination_city']} "
                f"is on track and expected around {eta}. We'll let you know if anything changes.")
    reason = customer_reason(risky[0]) if risky else "conditions on the route"
    late_by = max(pred.get("expected_delay_hours", 0), 0)
    if pred["risk_level"] == "High":
        return (f"Hi {customer_name}, shipment {sid} to {pred['destination_city']} is likely to be delayed, mainly "
                f"due to {reason}. Our current estimate is {eta} (about {late_by:.0f} hrs after the planned "
                f"{planned}). Our team is monitoring it closely and will update you if the estimate changes.")
    return (f"Hi {customer_name}, shipment {sid} to {pred['destination_city']} is expected around {eta}. "
            f"There is a moderate chance of delay because of {reason}; we're keeping an eye on it and will "
            "update you if anything changes.")


def llm_alert(pred: dict, customer_name: str = "Customer", tone: str = "friendly and professional") -> str:
    import anthropic                                   # imported lazily: optional dependency
    facts = {
        "customer_name": customer_name, "shipment_id": pred.get("shipment_id"),
        "route": f"{pred['origin_city']} to {pred['destination_city']}",
        "expected_arrival": _fmt_time(pred["expected_arrival"]),
        "planned_arrival": _fmt_time(pred["planned_arrival"]),
        "delay_risk": f"{pred['risk_level']} ({pred['delay_probability']:.0%})",
        "main_reasons": [customer_reason(r) for r in pred.get("top_reasons", [])
                         if r.get("risk_direction") == "raises risk"][:2],
    }
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=DEFAULT_MODEL, max_tokens=300, system=SYSTEM_PROMPT.format(tone=tone),
        messages=[{"role": "user", "content": f"Facts: {facts}\nWrite the customer message."}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def make_alert(pred: dict, customer_name: str = "Customer", tone: str = "friendly and professional",
               use_llm: bool | None = None) -> dict:
    """Return {"message", "mode"}; falls back to the template on any LLM problem."""
    want_llm = use_llm if use_llm is not None else bool(os.getenv("ANTHROPIC_API_KEY"))
    if want_llm:
        try:
            return {"message": llm_alert(pred, customer_name, tone), "mode": "llm"}
        except Exception as exc:                       # noqa: BLE001 - never break the demo
            log.warning("LLM alert failed (%s) -> template", exc)
    return {"message": template_alert(pred, customer_name), "mode": "template"}

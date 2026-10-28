"""Customer-facing delay messages.

``make_alert`` returns a message in one of two modes:

* ``llm``: an OpenAI-compatible chat-completions endpoint rephrases the
  prediction. Enabled when ``ARRIVO_LLM_API_KEY`` is set; ``ARRIVO_LLM_BASE_URL``
  and ``ARRIVO_LLM_MODEL`` select the provider and model.
* ``template``: deterministic text, used by default and as the fallback.

The model only receives facts from the prediction and is instructed not to
add causes, compensation or guarantees.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime

import requests

log = logging.getLogger(__name__)

LLM_TIMEOUT_S = 20

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
    """Strip parenthesised details and lower-case the first letter."""
    core = re.sub(r"\s*\([^)]*\)", "", text).strip()
    return core[0].lower() + core[1:] if core else core


# Customer-facing wording for each reason group.
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
    facts = {
        "customer_name": customer_name, "shipment_id": pred.get("shipment_id"),
        "route": f"{pred['origin_city']} to {pred['destination_city']}",
        "expected_arrival": _fmt_time(pred["expected_arrival"]),
        "planned_arrival": _fmt_time(pred["planned_arrival"]),
        "delay_risk": f"{pred['risk_level']} ({pred['delay_probability']:.0%})",
        "main_reasons": [customer_reason(r) for r in pred.get("top_reasons", [])
                         if r.get("risk_direction") == "raises risk"][:2],
    }
    base_url = os.getenv("ARRIVO_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['ARRIVO_LLM_API_KEY']}"},
        json={"model": os.getenv("ARRIVO_LLM_MODEL", "gpt-4o-mini"), "max_tokens": 300,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT.format(tone=tone)},
                           {"role": "user", "content": f"Facts: {facts}\nWrite the customer message."}]},
        timeout=LLM_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def make_alert(pred: dict, customer_name: str = "Customer", tone: str = "friendly and professional",
               use_llm: bool | None = None) -> dict:
    """Return ``{"message", "mode"}``, falling back to the template if the LLM call fails."""
    want_llm = use_llm if use_llm is not None else bool(os.getenv("ARRIVO_LLM_API_KEY"))
    if want_llm:
        try:
            return {"message": llm_alert(pred, customer_name, tone), "mode": "llm"}
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM alert failed, using template: %s", exc)
    return {"message": template_alert(pred, customer_name), "mode": "template"}

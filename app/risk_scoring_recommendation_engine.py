from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

DURATION_POINTS = {"short": 8, "medium": 22, "long": 38, "very_long": 44}
RISK_THRESHOLDS = {"low": 30, "medium": 55, "high": 75}
CONFIDENCE_WEIGHT = {
    "high": 1.0,
    "medium": 0.85,
    "low": 0.70,
}

CAUSE_OVERRIDES = {
    "vehicle_breakdown": {
        "risk_bonus": 8,
        "equipment": ["tow van", "clearance crew"],
        "agency": ["traffic police tow unit"],
    },
    "accident": {
        "risk_bonus": 14,
        "equipment": ["ambulance standby", "tow van", "crash cones"],
        "agency": ["traffic police", "medical emergency support"],
    },
    "tree_fall": {
        "risk_bonus": 16,
        "equipment": ["barricades", "tree clearance crew"],
        "agency": ["traffic police", "BBMP tree clearance"],
    },
    "water_logging": {
        "risk_bonus": 14,
        "equipment": ["barricades", "water pump support"],
        "agency": ["traffic police", "BBMP/BWSSB"],
    },
    "public_event": {
        "risk_bonus": 12,
        "equipment": ["barricades", "crowd-control ropes", "portable signage"],
        "agency": ["traffic police", "local law and order unit"],
    },
    "procession": {
        "risk_bonus": 12,
        "equipment": ["barricades", "route marshals", "portable signage"],
        "agency": ["traffic police", "local law and order unit"],
    },
    "vip_movement": {
        "risk_bonus": 18,
        "equipment": ["barricades", "pilot route signage"],
        "agency": ["traffic police", "VIP movement coordination cell"],
    },
    "construction": {
        "risk_bonus": 8,
        "equipment": ["barricades", "reflective cones", "warning signage"],
        "agency": ["traffic police", "road works contractor"],
    },
    "congestion": {
        "risk_bonus": 6,
        "equipment": ["traffic cones", "portable signage"],
        "agency": ["traffic police"],
    },
    "pot_holes": {
        "risk_bonus": 5,
        "equipment": ["reflective cones", "warning signage"],
        "agency": ["traffic police", "BBMP road maintenance"],
    },
}

BASE_PLAYBOOK = {
    "low": {
        "manpower": "1 patrol unit",
        "barricading": "No barricading by default; keep cones ready if lane obstruction grows.",
        "diversion": "No diversion. Monitor junction approach roads.",
        "control_room": "Log and monitor.",
    },
    "medium": {
        "manpower": "2 officers + 1 patrol unit",
        "barricading": "Local channelization with cones/signage near incident point.",
        "diversion": "Prepare local diversion if queue spills to adjacent junction.",
        "control_room": "Notify station control and monitor every 15 minutes.",
    },
    "high": {
        "manpower": "4-6 officers + 2 patrol units",
        "barricading": "Barricade readiness at affected arm and upstream junction.",
        "diversion": "Activate partial diversion and push advisory to nearby corridors.",
        "control_room": "Control-room loop with field updates every 10 minutes.",
    },
    "critical": {
        "manpower": "8+ officers + inspector oversight + 3 patrol units",
        "barricading": "Full barricade deployment and protected emergency lane.",
        "diversion": "Activate full diversion plan and public advisory immediately.",
        "control_room": "Dedicated control-room monitoring until clearance.",
    },
}


def _ensure_text_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str).str.strip().str.lower().replace({"": "unknown"})
    return out


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"


def _identify_cause_key(cause_text: Any) -> str:
    cause = _normalize_text(cause_text)
    for key in CAUSE_OVERRIDES:
        if key in cause:
            return key
    return "default"


def _risk_level_from_score(score: float) -> str:
    if score >= RISK_THRESHOLDS["high"]:
        return "critical"
    if score >= RISK_THRESHOLDS["medium"]:
        return "high"
    if score >= RISK_THRESHOLDS["low"]:
        return "medium"
    return "low"


def _bump_level(level: str, steps: int = 1) -> str:
    order = ["low", "medium", "high", "critical"]
    idx = order.index(level)
    idx = max(0, min(idx + steps, len(order) - 1))
    return order[idx]


def _confidence_bucket(conf: float) -> str:
    if conf >= 0.70:
        return "high"
    if conf >= 0.50:
        return "medium"
    return "low"


def build_recommendation(row: pd.Series) -> dict[str, Any]:
    closure_probability = float(row.get("road_closure_probability", 0.25))
    duration_band = _normalize_text(row.get("predicted_duration_band", "unknown"))
    cause_key = _identify_cause_key(row.get("event_cause", "unknown"))
    hotspot_score = float(row.get("hotspot_score", 0) or 0)
    peak_bonus = 5 if int(row.get("is_peak_hour", 0) or 0) == 1 else 0
    blocked_bonus = 4 if int(row.get("has_blocked_word", 0) or 0) == 1 else 0
    cause_bonus = CAUSE_OVERRIDES.get(cause_key, {}).get("risk_bonus", 0)

    dur_confidence = float(row.get("prediction_confidence", 0.5))
    conf_bucket = _confidence_bucket(dur_confidence)
    conf_weight = CONFIDENCE_WEIGHT[conf_bucket]
    duration_contribution = DURATION_POINTS.get(duration_band, 8) * conf_weight

    score = (
        closure_probability * 45
        + duration_contribution
        + min(hotspot_score, 40) * 0.30
        + peak_bonus
        + blocked_bonus
        + cause_bonus
    )
    score = float(np.clip(score, 0, 100))
    level = _risk_level_from_score(score)

    if duration_band in {"long", "very_long"} and closure_probability >= 0.60:
        level = _bump_level(level, 1)
    if cause_key in {"accident", "tree_fall", "vip_movement"} and closure_probability >= 0.50:
        level = _bump_level(level, 1)
    if conf_bucket == "low" and closure_probability < 0.30 and level in {"high", "critical"}:
        level = _bump_level(level, -1)

    base = BASE_PLAYBOOK[level]
    override = CAUSE_OVERRIDES.get(cause_key, {})

    equipment = sorted(set(["traffic cones", "reflective jackets"] + override.get("equipment", [])))
    agencies = sorted(set(["traffic police"] + override.get("agency", [])))

    if closure_probability >= 0.70 or level in {"high", "critical"}:
        equipment = sorted(set(equipment + ["barricades", "portable signage"]))
    if duration_band in {"long", "very_long"}:
        agencies = sorted(set(agencies + ["control room"]))

    return {
        "risk_level": level,
        "risk_score": round(score, 2),
        "confidence_bucket": conf_bucket,
        "manpower": base["manpower"],
        "barricading": base["barricading"],
        "diversion": base["diversion"],
        "control_room": base["control_room"],
        "equipment": equipment,
        "agency_alerts": agencies,
        "primary_trigger": cause_key,
    }


def build_risk_playbooks(history_df: pd.DataFrame, hotspot_summary: pd.DataFrame) -> pd.DataFrame:
    risk_df = history_df.copy()
    risk_df = _ensure_text_columns(risk_df, ["corridor", "junction", "police_station", "event_cause"])

    merge_cols = ["corridor", "junction", "police_station", "total_events", "closure_rate", "hotspot_score", "hotspot_level"]
    risk_df = risk_df.merge(hotspot_summary[merge_cols], on=["corridor", "junction", "police_station"], how="left")
    risk_df[["total_events", "closure_rate", "hotspot_score"]] = risk_df[["total_events", "closure_rate", "hotspot_score"]].fillna(0)
    risk_df["hotspot_level"] = risk_df["hotspot_level"].fillna("low")

    playbooks = risk_df.apply(build_recommendation, axis=1)
    playbook_df = pd.json_normalize(playbooks)

    risk_playbooks = pd.concat([risk_df.reset_index(drop=True), playbook_df], axis=1)
    risk_playbooks["equipment"] = risk_playbooks["equipment"].apply(lambda items: ", ".join(items))
    risk_playbooks["agency_alerts"] = risk_playbooks["agency_alerts"].apply(lambda items: ", ".join(items))
    risk_playbooks["playbook_json"] = playbooks.apply(json.dumps)

    return risk_playbooks


def build_recommendation_summary(
    risk_playbooks: pd.DataFrame,
    hotspot_summary: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "events_scored",
                "low_risk_events",
                "medium_risk_events",
                "high_risk_events",
                "critical_risk_events",
                "avg_road_closure_probability",
                "avg_risk_score",
                "avg_prediction_confidence",
                "high_confidence_pct",
                "medium_confidence_pct",
                "low_confidence_pct",
                "hotspot_rows",
                "cause_overrides_count",
            ],
            "value": [
                len(risk_playbooks),
                int(risk_playbooks["risk_level"].eq("low").sum()),
                int(risk_playbooks["risk_level"].eq("medium").sum()),
                int(risk_playbooks["risk_level"].eq("high").sum()),
                int(risk_playbooks["risk_level"].eq("critical").sum()),
                round(float(risk_playbooks["road_closure_probability"].mean()), 4),
                round(float(risk_playbooks["risk_score"].mean()), 4),
                round(float(risk_playbooks["prediction_confidence"].mean()), 4),
                round(float(risk_playbooks["confidence_bucket"].eq("high").mean() * 100), 2),
                round(float(risk_playbooks["confidence_bucket"].eq("medium").mean() * 100), 2),
                round(float(risk_playbooks["confidence_bucket"].eq("low").mean() * 100), 2),
                len(hotspot_summary),
                len(CAUSE_OVERRIDES),
            ],
        }
    )

from __future__ import annotations

import math
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from .hotspot_analytics import build_heatmap_points, build_hotspot_summary
    from .risk_scoring_recommendation_engine import (
        build_recommendation,
        build_recommendation_summary,
        build_risk_playbooks,
    )
    from .similarity_incident_retrieval import build_similarity_store, query_similar_incidents
except ImportError:
    from hotspot_analytics import build_heatmap_points, build_hotspot_summary
    from risk_scoring_recommendation_engine import (
        build_recommendation,
        build_recommendation_summary,
        build_risk_playbooks,
    )
    from similarity_incident_retrieval import build_similarity_store, query_similar_incidents


REQUIRED_EVENT_FIELDS = [
    "event_type",
    "event_cause",
    "start_datetime",
    "created_date",
    "latitude",
    "longitude",
    "corridor",
    "police_station",
    "junction",
    "zone",
    "veh_type",
    "description",
]


@dataclass
class PipelineArtifacts:
    history_df: pd.DataFrame
    risk_playbooks: pd.DataFrame
    hotspot_summary: pd.DataFrame
    recommendation_summary: pd.DataFrame
    similarity_meta: pd.DataFrame
    similarity_vectors: np.ndarray
    similarity_index: NearestNeighbors
    scaler: StandardScaler
    text_pca: PCA | None
    struct_cols: list[str]
    model1_bundle: dict[str, Any] | None
    model2_artifact: dict[str, Any] | None


@dataclass
class LiveWorkflowResult:
    event_row: pd.DataFrame
    recommendation_row: pd.DataFrame
    similar_incidents: pd.DataFrame
    model1_prediction: dict[str, Any]
    model2_prediction: dict[str, Any]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    return text if text else "unknown"


def _ensure_text_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str).str.strip().str.lower().replace({"": "unknown"})
    return out


def parse_event_datetime(value: Any) -> datetime | None:
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def sanitize_event_input(raw_event: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED_EVENT_FIELDS if k not in raw_event]
    if missing:
        raise ValueError(f"Missing required event fields: {missing}")

    event = {k: raw_event.get(k) for k in REQUIRED_EVENT_FIELDS}
    for col in ["event_type", "event_cause", "corridor", "police_station", "junction", "zone", "veh_type"]:
        event[col] = normalize_text(event.get(col))

    event["description"] = str(event.get("description") or "").strip()
    event["latitude"] = float(event.get("latitude"))
    event["longitude"] = float(event.get("longitude"))

    start_dt = parse_event_datetime(event.get("start_datetime"))
    created_dt = parse_event_datetime(event.get("created_date"))
    if start_dt is None:
        start_dt = datetime.now()
    if created_dt is None:
        created_dt = start_dt

    event["start_datetime"] = start_dt.isoformat(timespec="minutes")
    event["created_date"] = created_dt.isoformat(timespec="minutes")
    return event


def text_risk_flags(description: str) -> dict[str, bool]:
    text = str(description or "").lower()
    blocked_terms = ["blocked", "block", "closed", "closure", "obstruction", "stuck"]
    jam_terms = ["jam", "congestion", "heavy traffic", "slow traffic", "queue"]
    severity_terms = ["major", "severe", "huge", "massive", "critical", "urgent"]
    diversion_terms = ["diversion", "divert", "reroute"]
    public_terms = ["rally", "protest", "procession", "festival", "vip", "public", "gathering"]
    return {
        "has_blocked_word": any(term in text for term in blocked_terms),
        "has_jam_word": any(term in text for term in jam_terms),
        "has_severity_word": any(term in text for term in severity_terms),
        "has_diversion_word": any(term in text for term in diversion_terms),
        "has_public_event_word": any(term in text for term in public_terms),
    }


def haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = lat2.map(math.radians)
    d_phi = (lat2 - lat1).map(math.radians)
    d_lam = (lon2 - lon1).map(math.radians)
    a = (d_phi / 2).map(math.sin) ** 2 + math.cos(phi1) * phi2.map(math.cos) * (d_lam / 2).map(math.sin) ** 2
    return 2 * radius * a.map(lambda value: math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1 - value))))


def _choose_feature_template(history_df: pd.DataFrame, event: dict[str, Any]) -> pd.Series:
    candidates = history_df.copy()
    if "latitude" in candidates.columns and "longitude" in candidates.columns:
        candidates["distance_km"] = haversine_km(
            float(event["latitude"]),
            float(event["longitude"]),
            pd.to_numeric(candidates["latitude"], errors="coerce").fillna(float(event["latitude"])),
            pd.to_numeric(candidates["longitude"], errors="coerce").fillna(float(event["longitude"])),
        )
    else:
        candidates["distance_km"] = 0.0

    candidates["match_score"] = 0.0
    for column, weight in [
        ("event_cause", 2.0),
        ("corridor", 1.5),
        ("police_station", 1.0),
        ("junction", 1.0),
        ("zone", 0.9),
        ("veh_type", 0.8),
        ("event_type", 0.75),
    ]:
        if column in candidates.columns:
            candidates[column] = candidates[column].fillna("unknown").astype(str).str.lower().str.strip()
            candidates["match_score"] += (candidates[column] == str(event[column])).astype(float) * weight

    candidates["similarity_score"] = candidates["match_score"] - candidates["distance_km"].clip(0, 20) / 8
    return candidates.sort_values(["similarity_score", "distance_km"], ascending=[False, True]).iloc[0].copy()


def _build_feature_row(history_df: pd.DataFrame, event: dict[str, Any]) -> pd.Series:
    row = _choose_feature_template(history_df, event)
    center_lat, center_lon = 12.9716, 77.5946

    start_dt = parse_event_datetime(event["start_datetime"]) or datetime.now()
    created_dt = parse_event_datetime(event["created_date"]) or start_dt
    flags = text_risk_flags(event["description"])
    text = str(event["description"] or "")

    row["latitude"] = float(event["latitude"])
    row["longitude"] = float(event["longitude"])
    row["lat_round_3"] = round(float(event["latitude"]), 3)
    row["lon_round_3"] = round(float(event["longitude"]), 3)
    row["distance_to_city_center_km"] = float(
        haversine_km(float(event["latitude"]), float(event["longitude"]), pd.Series([center_lat]), pd.Series([center_lon])).iloc[0]
    )

    row["start_hour"] = int(start_dt.hour)
    row["start_dayofweek"] = int(start_dt.weekday())
    row["start_month_number"] = int(start_dt.month)
    row["start_weekofyear"] = int(start_dt.isocalendar().week)
    row["is_weekend"] = int(start_dt.weekday() >= 5)
    row["is_morning_peak"] = int(start_dt.hour in {8, 9, 10})
    row["is_evening_peak"] = int(start_dt.hour in {17, 18, 19, 20})
    row["is_peak_hour"] = int(row["is_morning_peak"] or row["is_evening_peak"])
    row["is_night"] = int(start_dt.hour >= 22 or start_dt.hour <= 5)
    row["hour_sin"] = math.sin(2 * math.pi * start_dt.hour / 24)
    row["hour_cos"] = math.cos(2 * math.pi * start_dt.hour / 24)
    row["day_sin"] = math.sin(2 * math.pi * start_dt.weekday() / 7)
    row["day_cos"] = math.cos(2 * math.pi * start_dt.weekday() / 7)

    report_lag = max(0.0, (start_dt - created_dt).total_seconds() / 60)
    row["report_lag_minutes_clipped"] = min(report_lag, 24 * 60)
    row["report_lag_hours_clipped"] = min(report_lag / 60, 24)
    row["report_lag_missing"] = 0

    row["description_missing"] = int(not text.strip())
    row["text_length"] = len(text)
    row["description_char_length"] = len(text)
    row["description_word_count"] = len(text.split())
    row["has_non_ascii_text"] = int(any(ord(char) > 127 for char in text))
    row["has_kannada_text"] = int(any("\u0c80" <= char <= "\u0cff" for char in text))
    row["has_accident_word"] = int("accident" in text.lower() or "crash" in text.lower())
    row["has_breakdown_word"] = int("breakdown" in text.lower())
    row["has_water_word"] = int("water" in text.lower() or "flood" in text.lower() or "rain" in text.lower())
    row["has_construction_word"] = int("construction" in text.lower() or "work" in text.lower())
    row["has_event_word"] = int(flags["has_public_event_word"])
    row["has_blocked_word"] = int(flags["has_blocked_word"])
    row["has_jam_word"] = int(flags["has_jam_word"])
    row["has_vip_word"] = int("vip" in text.lower())
    row["has_location_hint_word"] = int(any(term in text.lower() for term in ["near", "junction", "road", "circle", "signal"]))

    row["is_planned_event"] = int(event["event_type"] == "planned")
    row["is_public_or_vip_event"] = int(flags["has_public_event_word"])
    row["is_breakdown_event"] = int(event["event_cause"] == "vehicle_breakdown" or row["has_breakdown_word"])
    row["is_accident_event"] = int(event["event_cause"] == "accident" or row["has_accident_word"])
    row["is_weather_or_visibility_event"] = int(row["has_water_word"] or "weather" in str(event["event_cause"]))
    row["is_road_condition_event"] = int(any(term in str(event["event_cause"]) for term in ["road", "pothole", "water"]))
    row["has_vehicle_type"] = int(event["veh_type"] != "unknown")
    row["is_truck"] = int("truck" in str(event["veh_type"]).lower())
    row["is_bus"] = int("bus" in str(event["veh_type"]).lower())
    row["is_heavy_vehicle"] = int(row["is_truck"] or row["is_bus"] or "heavy" in str(event["veh_type"]).lower())

    peak_period = "morning_peak" if row["is_morning_peak"] else "evening_peak" if row["is_evening_peak"] else "night" if row["is_night"] else "off_peak"

    row["event_type"] = event["event_type"]
    row["event_cause"] = event["event_cause"]
    row["veh_type"] = event["veh_type"]
    row["corridor"] = event["corridor"]
    row["police_station"] = event["police_station"]
    row["zone"] = event["zone"]
    row["junction"] = event["junction"]
    row["description"] = text
    row["start_datetime"] = start_dt.isoformat(timespec="minutes")
    row["created_date"] = created_dt.isoformat(timespec="minutes")

    row["location_grid"] = f"{row['lat_round_3']:.3f}_{row['lon_round_3']:.3f}"
    row["peak_period"] = peak_period
    row["start_day_name"] = start_dt.strftime("%A").lower()
    row["start_month_name"] = start_dt.strftime("%B").lower()
    row["event_cause_corridor"] = f"{event['event_cause']}_{event['corridor']}"
    row["cause_peak_interaction"] = f"{event['event_cause']}_{peak_period}"
    row["zone_cause_interaction"] = f"{event['zone']}_{event['event_cause']}"
    row["corridor_cause_interaction"] = f"{event['corridor']}_{event['event_cause']}"
    row["cause_heavy_vehicle_interaction"] = f"{event['event_cause']}_heavy_{row['is_heavy_vehicle']}"
    row["corridor_peak_interaction"] = f"{event['corridor']}_{peak_period}"
    row["cause_hour_interaction"] = f"{event['event_cause']}_{start_dt.hour}"
    row["vehicle_cause_interaction"] = f"{event['veh_type']}_{event['event_cause']}"
    row["cargo_vehicle_interaction"] = f"{event['veh_type']}_{event['veh_type']}"
    row["planned_peak_interaction"] = f"{event['event_type']}_{peak_period}"
    row["weather_zone_interaction"] = f"weather_{row['is_weather_or_visibility_event']}_{event['zone']}"

    if "id" not in row.index:
        row["id"] = f"LIVE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return row


def load_model1_bundle(project_root: Path) -> dict[str, Any]:
    bundle_path = project_root / "outputs" / "model1_road_closure" / "model1_inference_bundle.pkl"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing Model 1 bundle: {bundle_path}")
    return joblib.load(bundle_path)


def load_model2_artifact(project_root: Path) -> dict[str, Any]:
    candidate_paths = [
        project_root / "outputs" / "model2_duration_band" / "model2_v1_duration_band_model.pkl",
        project_root / "models" / "model2_v1_duration_band_model.pkl",
    ]
    for path in candidate_paths:
        if path.exists():
            return joblib.load(path)
    raise FileNotFoundError(f"Missing Model 2 artifact in any expected path: {candidate_paths}")


def predict_model1_closure(feature_row: pd.Series, model1_bundle: dict[str, Any]) -> dict[str, Any]:
    model_input_cols = model1_bundle["model_input_cols"]
    encoded_feature_cols = model1_bundle["encoded_feature_cols"]
    preprocessor = model1_bundle["preprocessor"]
    portable = model1_bundle["model"]

    x_raw = pd.DataFrame([feature_row]).reindex(columns=model_input_cols)
    x_enc = preprocessor.transform(x_raw)
    x_enc_df = pd.DataFrame(x_enc, columns=encoded_feature_cols)

    lgb_raw = np.asarray(
        portable["lgb_booster"].predict(x_enc_df, num_iteration=int(portable["lgb_best_iteration"]))
    ).reshape(-1)
    xgb_raw = np.asarray(
        portable["xgb_booster"].predict(
            xgb.DMatrix(x_enc_df.to_numpy(), feature_names=list(x_enc_df.columns)),
            iteration_range=(0, int(portable["xgb_best_iteration"]) + 1),
        )
    ).reshape(-1)

    lgb_cal = portable["lgb_calibrator"].predict_proba(lgb_raw.reshape(-1, 1))[:, 1]
    xgb_cal = portable["xgb_calibrator"].predict_proba(xgb_raw.reshape(-1, 1))[:, 1]

    lgb_weight = float(portable.get("lgb_weight", model1_bundle.get("lgb_weight", 0.5)))
    xgb_weight = float(portable.get("xgb_weight", model1_bundle.get("xgb_weight", 0.5)))
    closure_probability = float(np.clip(lgb_weight * lgb_cal[0] + xgb_weight * xgb_cal[0], 0, 1))

    threshold = float(model1_bundle.get("optimal_threshold", 0.5))
    label = int(closure_probability >= threshold)
    return {
        "road_closure_probability": closure_probability,
        "road_closure_label": label,
        "model1_threshold": threshold,
        "lgb_calibrated_probability": float(lgb_cal[0]),
        "xgb_calibrated_probability": float(xgb_cal[0]),
        "model_source": "model1_inference_bundle",
    }


def predict_model2_duration_from_history(
    feature_row: pd.Series,
    history_df: pd.DataFrame,
    closure_probability: float,
    top_k: int = 30,
) -> dict[str, Any]:
    candidates = history_df.copy()
    candidates = _ensure_text_columns(candidates, ["event_type", "event_cause", "corridor", "police_station", "junction", "zone", "veh_type"])

    candidates["distance_km"] = haversine_km(
        float(feature_row.get("latitude", 12.9716)),
        float(feature_row.get("longitude", 77.5946)),
        pd.to_numeric(candidates.get("latitude", pd.Series(12.9716, index=candidates.index)), errors="coerce").fillna(12.9716),
        pd.to_numeric(candidates.get("longitude", pd.Series(77.5946, index=candidates.index)), errors="coerce").fillna(77.5946),
    )
    candidates["match_score"] = 0.0
    for col, wt in [
        ("event_cause", 2.0),
        ("corridor", 1.5),
        ("police_station", 1.0),
        ("junction", 1.0),
        ("zone", 0.9),
        ("veh_type", 0.8),
        ("event_type", 0.75),
    ]:
        if col in candidates.columns:
            candidates["match_score"] += (candidates[col] == normalize_text(feature_row.get(col, "unknown"))).astype(float) * wt

    candidates["closure_gap"] = (pd.to_numeric(candidates.get("road_closure_probability", 0.25), errors="coerce").fillna(0.25) - closure_probability).abs()
    candidates["similarity_score"] = candidates["match_score"] - candidates["distance_km"].clip(0, 20) / 8 - candidates["closure_gap"] * 2.5

    nearest = candidates.sort_values(["similarity_score", "distance_km"], ascending=[False, True]).head(max(1, top_k)).copy()
    weights = (1 / (1 + nearest["distance_km"].clip(lower=0))) * (1 + nearest["match_score"].clip(lower=0))
    weights = weights / max(float(weights.sum()), 1e-9)

    prob_cols = [c for c in nearest.columns if c.startswith("prob_")]
    duration_labels = ["short", "medium", "long", "very_long"]
    duration_probs = {k: 0.0 for k in duration_labels}

    if prob_cols:
        for col in prob_cols:
            if col.startswith("prob_duration_"):
                label = col.replace("prob_duration_", "")
            else:
                label = col.replace("prob_", "")
            if label in duration_probs:
                duration_probs[label] += float((pd.to_numeric(nearest[col], errors="coerce").fillna(0) * weights).sum())

        total = sum(duration_probs.values())
        if total > 0:
            duration_probs = {k: float(v / total) for k, v in duration_probs.items()}
        else:
            prob_cols = []

    if not prob_cols:
        band_series = nearest.get("predicted_duration_band", pd.Series("medium", index=nearest.index)).astype(str).str.lower()
        for label in duration_labels:
            duration_probs[label] = float(weights[band_series.eq(label)].sum())
        total = sum(duration_probs.values())
        if total <= 0:
            duration_probs = {"short": 0.1, "medium": 0.7, "long": 0.15, "very_long": 0.05}
        else:
            duration_probs = {k: float(v / total) for k, v in duration_probs.items()}

    predicted_duration_band = max(duration_probs, key=duration_probs.get)
    prediction_confidence = float(max(duration_probs.values()))

    return {
        "predicted_duration_band": predicted_duration_band,
        "prediction_confidence": prediction_confidence,
        "duration_probs": duration_probs,
        "model_source": "historical_similarity_proxy",
    }


def predict_model2_duration_from_model(
    feature_row: pd.Series,
    model2_artifact: dict[str, Any],
    closure_probability: float,
) -> dict[str, Any]:
    model = model2_artifact["model"]
    feature_cols = list(model2_artifact.get("feature_cols", []))
    label_encoder = model2_artifact["label_encoder"]

    if not feature_cols:
        raise ValueError("Model 2 artifact missing feature_cols.")

    x_row = pd.DataFrame([feature_row]).copy()
    # Support either naming convention from Model 2 training notebook.
    x_row["road_closure_probability"] = float(closure_probability)
    x_row["model1_closure_probability"] = float(closure_probability)

    x_model = x_row.reindex(columns=feature_cols)
    for col in x_model.columns:
        x_model[col] = pd.to_numeric(x_model[col], errors="coerce")
    x_model = x_model.replace([np.inf, -np.inf], np.nan)
    x_model = x_model.fillna(0.0)

    proba = np.asarray(model.predict_proba(x_model))
    pred = np.asarray(model.predict(x_model)).astype(int).reshape(-1)

    class_labels = [str(c) for c in label_encoder.classes_]
    probs = proba[0] if proba.ndim == 2 and len(proba) else np.zeros(len(class_labels), dtype=float)
    duration_probs = {label: float(prob) for label, prob in zip(class_labels, probs)}

    # Ensure all expected classes are present for downstream UI/heatmap logic.
    for label in ["short", "medium", "long", "very_long"]:
        duration_probs.setdefault(label, 0.0)

    total = float(sum(duration_probs.values()))
    if total > 0:
        duration_probs = {k: float(v / total) for k, v in duration_probs.items()}

    predicted_duration_band = str(label_encoder.inverse_transform(pred)[0])
    prediction_confidence = float(max(duration_probs.values())) if duration_probs else 0.0

    return {
        "predicted_duration_band": predicted_duration_band,
        "prediction_confidence": prediction_confidence,
        "duration_probs": duration_probs,
        "model_source": "model2_v1_duration_band_model",
    }


def build_live_event_record(
    event: dict[str, Any],
    feature_row: pd.Series,
    model1_pred: dict[str, Any],
    model2_pred: dict[str, Any],
) -> dict[str, Any]:
    row = feature_row.copy()
    row["id"] = f"LIVE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for k, v in event.items():
        row[k] = v

    row["road_closure_probability"] = float(model1_pred["road_closure_probability"])
    row["target_road_closure"] = int(model1_pred["road_closure_label"])
    row["predicted_duration_band"] = str(model2_pred["predicted_duration_band"])
    row["prediction_confidence"] = float(model2_pred["prediction_confidence"])

    for label in ["short", "medium", "long", "very_long"]:
        row[f"prob_{label}"] = float(model2_pred["duration_probs"].get(label, 0.0))

    return row.to_dict()


def _load_historical_df(project_root: Path) -> pd.DataFrame:
    handoff_candidates = [
        project_root / "outputs" / "model1_road_closure" / "model2_duration_handoff.csv",
        project_root / "outputs" / "model_road_closure" / "model2_duration_handoff.csv",  # backward compatibility
    ]
    model1_features = project_root / "outputs" / "features" / "road_closure_features_v1.csv"
    model2_handoff = next((p for p in handoff_candidates if p.exists()), None)

    if model2_handoff is not None:
        df = pd.read_csv(model2_handoff, low_memory=False)
    elif model1_features.exists():
        df = pd.read_csv(model1_features, low_memory=False)
    else:
        raise FileNotFoundError(
            "Missing historical inputs under outputs/features or outputs/model1_road_closure."
        )

    pred_candidates = [
        project_root / "outputs" / "model2_duration_band" / "model2_v1_duration_band_predictions.csv",
        project_root / "outputs" / "model2_v1_duration_band_predictions.csv",  # backward compatibility
    ]
    pred_path = next((p for p in pred_candidates if p.exists()), None)
    if pred_path is not None and "_source_row" in df.columns:
        pred_df = pd.read_csv(pred_path, low_memory=False)
        if "source_row_index" in pred_df.columns:
            keep_cols = ["source_row_index", "predicted_duration_band", "prediction_confidence"]
            keep_cols += [c for c in pred_df.columns if c.startswith("prob_")]
            keep_cols = [c for c in keep_cols if c in pred_df.columns]
            df = df.merge(pred_df[keep_cols], left_on="_source_row", right_on="source_row_index", how="left")

    if "predicted_duration_band" not in df.columns:
        fallback_band = df["duration_band"] if "duration_band" in df.columns else "unknown"
        df["predicted_duration_band"] = fallback_band

    if "prediction_confidence" not in df.columns:
        df["prediction_confidence"] = 0.50

    if "road_closure_probability" not in df.columns:
        df["road_closure_probability"] = 0.25

    df["prediction_confidence"] = pd.to_numeric(df["prediction_confidence"], errors="coerce").fillna(0.50).clip(0, 1)
    df["road_closure_probability"] = pd.to_numeric(df["road_closure_probability"], errors="coerce").fillna(0.25).clip(0, 1)
    return df


def run_full_pipeline(project_root: Path, persist_outputs: bool = True) -> PipelineArtifacts:
    history_df = _load_historical_df(project_root)
    model1_bundle: dict[str, Any] | None = None
    model2_artifact: dict[str, Any] | None = None
    try:
        model1_bundle = load_model1_bundle(project_root)
    except FileNotFoundError:
        model1_bundle = None
    try:
        model2_artifact = load_model2_artifact(project_root)
    except FileNotFoundError:
        model2_artifact = None
    hotspot_summary = build_hotspot_summary(history_df)
    risk_playbooks = build_risk_playbooks(history_df, hotspot_summary)
    recommendation_summary = build_recommendation_summary(risk_playbooks, hotspot_summary)

    (
        similarity_meta,
        similarity_vectors,
        similarity_index,
        scaler,
        text_pca,
        struct_cols,
    ) = build_similarity_store(risk_playbooks)

    if persist_outputs:
        rec_dir = project_root / "outputs" / "recommendations"
        sim_dir = project_root / "outputs" / "similarity"
        rec_dir.mkdir(parents=True, exist_ok=True)
        sim_dir.mkdir(parents=True, exist_ok=True)

        risk_playbooks.to_csv(rec_dir / "event_risk_playbooks.csv", index=False)
        hotspot_summary.to_csv(rec_dir / "hotspot_summary.csv", index=False)
        recommendation_summary.to_csv(rec_dir / "recommendation_summary.csv", index=False)

        pd.DataFrame(similarity_vectors).to_csv(sim_dir / "retrieval_vectors.csv", index=False)
        joblib.dump(
            {
                "retriever_type": "sklearn_nearestneighbors_cosine",
                "index": similarity_index,
                "scaler": scaler,
                "text_pca": text_pca,
                "struct_cols": struct_cols,
                "meta": similarity_meta,
            },
            sim_dir / "retrieval_artifacts.joblib",
        )

    return PipelineArtifacts(
        history_df=history_df,
        risk_playbooks=risk_playbooks,
        hotspot_summary=hotspot_summary,
        recommendation_summary=recommendation_summary,
        similarity_meta=similarity_meta,
        similarity_vectors=similarity_vectors,
        similarity_index=similarity_index,
        scaler=scaler,
        text_pca=text_pca,
        struct_cols=struct_cols,
        model1_bundle=model1_bundle,
        model2_artifact=model2_artifact,
    )


def score_current_event(current_event: dict[str, Any], artifacts: PipelineArtifacts) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_df = pd.DataFrame([current_event]).copy()
    event_df = _ensure_text_columns(event_df, ["corridor", "junction", "police_station", "event_cause"])

    # Remove any pre-existing recommendation outputs inherited from template rows.
    rec_cols = [
        "risk_level",
        "risk_score",
        "confidence_bucket",
        "manpower",
        "barricading",
        "diversion",
        "control_room",
        "equipment",
        "agency_alerts",
        "primary_trigger",
    ]
    for col in rec_cols:
        if col in event_df.columns:
            event_df = event_df.drop(columns=[col])

    for col in ["total_events", "closure_rate", "hotspot_score", "hotspot_level"]:
        if col in event_df.columns:
            event_df = event_df.drop(columns=[col])

    if "road_closure_probability" not in event_df.columns:
        event_df["road_closure_probability"] = 0.25
    if "predicted_duration_band" not in event_df.columns:
        event_df["predicted_duration_band"] = "medium"
    if "prediction_confidence" not in event_df.columns:
        event_df["prediction_confidence"] = 0.50

    event_df["road_closure_probability"] = pd.to_numeric(event_df["road_closure_probability"], errors="coerce").fillna(0.25).clip(0, 1)
    event_df["prediction_confidence"] = pd.to_numeric(event_df["prediction_confidence"], errors="coerce").fillna(0.50).clip(0, 1)

    event_df = event_df.merge(
        artifacts.hotspot_summary[
            ["corridor", "junction", "police_station", "total_events", "closure_rate", "hotspot_score", "hotspot_level"]
        ],
        on=["corridor", "junction", "police_station"],
        how="left",
    )
    event_df[["total_events", "closure_rate", "hotspot_score"]] = event_df[["total_events", "closure_rate", "hotspot_score"]].fillna(0)
    event_df["hotspot_level"] = event_df["hotspot_level"].fillna("low")

    rec = event_df.apply(build_recommendation, axis=1)
    rec_df = pd.concat([event_df.reset_index(drop=True), pd.json_normalize(rec)], axis=1)
    rec_df["equipment"] = rec_df["equipment"].apply(lambda x: ", ".join(x))
    rec_df["agency_alerts"] = rec_df["agency_alerts"].apply(lambda x: ", ".join(x))

    similar = query_similar_incidents(
        rec_df.iloc[0],
        similarity_meta=artifacts.similarity_meta,
        similarity_index=artifacts.similarity_index,
        scaler=artifacts.scaler,
        text_pca=artifacts.text_pca,
        struct_cols=artifacts.struct_cols,
        top_k=5,
    )
    return rec_df, similar


def run_live_event_workflow(
    project_root: Path,
    artifacts: PipelineArtifacts,
    raw_event: dict[str, Any],
) -> LiveWorkflowResult:
    event = sanitize_event_input(raw_event)
    feature_row = _build_feature_row(artifacts.risk_playbooks, event)

    model1_bundle = artifacts.model1_bundle if artifacts.model1_bundle is not None else load_model1_bundle(project_root)
    model1_pred = predict_model1_closure(feature_row, model1_bundle)

    model2_artifact = artifacts.model2_artifact
    if model2_artifact is None:
        try:
            model2_artifact = load_model2_artifact(project_root)
        except FileNotFoundError:
            model2_artifact = None

    if model2_artifact is not None:
        model2_pred = predict_model2_duration_from_model(
            feature_row,
            model2_artifact,
            closure_probability=float(model1_pred["road_closure_probability"]),
        )
    else:
        model2_pred = predict_model2_duration_from_history(
            feature_row,
            artifacts.risk_playbooks,
            closure_probability=float(model1_pred["road_closure_probability"]),
        )

    live_event_record = build_live_event_record(event, feature_row, model1_pred, model2_pred)
    rec_df, similar_df = score_current_event(live_event_record, artifacts)
    return LiveWorkflowResult(
        event_row=pd.DataFrame([live_event_record]),
        recommendation_row=rec_df,
        similar_incidents=similar_df,
        model1_prediction=model1_pred,
        model2_prediction=model2_pred,
    )

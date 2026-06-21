from __future__ import annotations

import json
import math
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_PATH = PROJECT_ROOT / "outputs" / "model_duration_band" / "model2_v1_duration_band_recommendation_handoff.csv"
MODEL2_FEATURE_PATH = PROJECT_ROOT / "outputs" / "model_road_closure" / "model2_duration_handoff.csv"
MODEL2_MODEL_PATH = PROJECT_ROOT / "outputs" / "model_duration_band" / "model2_v1_duration_band_model.pkl"
EVENT_LOG_PATH = PROJECT_ROOT / "outputs" / "live_events" / "event_recommendation_log.csv"
BENGALURU_CENTER = (12.9716, 77.5946)

DURATION_ORDER = ["short", "medium", "long", "very_long"]
RISK_COLORS = {
    "low": "#0f766e",
    "medium": "#ca8a04",
    "high": "#ea580c",
    "critical": "#dc2626",
}

st.set_page_config(page_title="EventOps AI", page_icon="🚦", layout="wide")


@st.cache_data
def load_handoff_data() -> pd.DataFrame:
    df = pd.read_csv(HANDOFF_PATH)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    for column in ["event_cause", "corridor", "police_station", "zone", "veh_type", "event_type", "junction"]:
        if column in df.columns:
            df[column] = df[column].fillna("unknown").astype(str)
    return df


@st.cache_data
def load_model2_feature_data() -> pd.DataFrame:
    df = pd.read_csv(MODEL2_FEATURE_PATH)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    for column in ["event_cause", "corridor", "police_station", "zone", "veh_type", "event_type", "junction"]:
        if column in df.columns:
            df[column] = df[column].fillna("unknown").astype(str)
    return df


@st.cache_resource
def load_model2_bundle() -> dict:
    return joblib.load(MODEL2_MODEL_PATH)


@st.cache_data(ttl=23 * 60 * 60, show_spinner=False)
def get_mappls_access_token(client_id: str, client_secret: str) -> dict:
    token_url = "https://outpost.mappls.com/api/security/oauth/token"
    payload = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode()
    request = urllib.request.Request(
        token_url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
            return {"ok": True, "access_token": data.get("access_token", ""), "expires_in": data.get("expires_in")}
    except urllib.error.HTTPError as exc:
        body = exc.read(300).decode("utf-8", errors="replace")
        return {"ok": False, "error": f"HTTP {exc.code}: {body}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def haversine_km(lat1: float, lon1: float, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = lat2.map(math.radians)
    d_phi = (lat2 - lat1).map(math.radians)
    d_lam = (lon2 - lon1).map(math.radians)
    a = (d_phi / 2).map(math.sin) ** 2 + math.cos(phi1) * phi2.map(math.cos) * (d_lam / 2).map(math.sin) ** 2
    return 2 * radius * a.map(lambda value: math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1 - value))))


def risk_level(score: float) -> str:
    if score >= 75:
        return "critical"
    if score >= 55:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def recommend_manpower(level: str, duration: str, closure_probability: float) -> dict:
    if level == "critical":
        manpower = "12-16 officers + rapid response unit"
        barricading = "Full barricading kit at primary and secondary approach roads"
        diversion = "Activate pre-approved diversion plan and broadcast alerts"
    elif level == "high":
        manpower = "8-10 officers"
        barricading = "Partial barricading at choke points"
        diversion = "Prepare diversion; activate if queue spillback starts"
    elif level == "medium":
        manpower = "4-6 officers"
        barricading = "Cones and mobile barricades on standby"
        diversion = "Monitor corridor; push advisory if congestion rises"
    else:
        manpower = "2-3 officers"
        barricading = "No fixed barricading; keep patrol mobile"
        diversion = "No diversion required"

    if duration in {"long", "very_long"}:
        diversion += "; keep plan active for extended duration"
    if closure_probability >= 0.65:
        barricading += "; include closure-control signage"

    return {
        "manpower": manpower,
        "barricading": barricading,
        "diversion": diversion,
        "alerts": "Traffic control room, nearest police station, civic response team",
    }


def text_risk_flags(description: str) -> dict:
    text = str(description).lower()
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


def parse_event_datetime(value: str) -> datetime | None:
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def choose_feature_template(feature_df: pd.DataFrame, event: dict) -> pd.Series:
    candidates = feature_df.copy()
    candidates["distance_km"] = haversine_km(
        event["latitude"],
        event["longitude"],
        candidates["latitude"],
        candidates["longitude"],
    )
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
            candidates["match_score"] += (candidates[column].astype(str) == str(event[column])).astype(float) * weight
    candidates["similarity_score"] = candidates["match_score"] - candidates["distance_km"].clip(0, 20) / 8
    return candidates.sort_values(["similarity_score", "distance_km"], ascending=[False, True]).iloc[0].copy()


def build_model2_feature_row(feature_df: pd.DataFrame, event: dict, closure_probability: float) -> pd.Series:
    row = choose_feature_template(feature_df, event)
    event_dt = parse_event_datetime(event["start_datetime"]) or datetime.now()
    created_dt = parse_event_datetime(event["created_date"]) or event_dt
    flags = text_risk_flags(event["description"])
    text = str(event["description"] or "")

    row["latitude"] = event["latitude"]
    row["longitude"] = event["longitude"]
    row["lat_round_3"] = round(event["latitude"], 3)
    row["lon_round_3"] = round(event["longitude"], 3)
    row["distance_to_city_center_km"] = float(
        haversine_km(event["latitude"], event["longitude"], pd.Series([BENGALURU_CENTER[0]]), pd.Series([BENGALURU_CENTER[1]])).iloc[0]
    )
    row["start_hour"] = event_dt.hour
    row["start_dayofweek"] = event_dt.weekday()
    row["start_month_number"] = event_dt.month
    row["start_weekofyear"] = int(event_dt.isocalendar().week)
    row["is_weekend"] = int(event_dt.weekday() >= 5)
    row["is_morning_peak"] = int(event_dt.hour in {8, 9, 10})
    row["is_evening_peak"] = int(event_dt.hour in {17, 18, 19, 20})
    row["is_peak_hour"] = int(row["is_morning_peak"] or row["is_evening_peak"])
    row["is_night"] = int(event_dt.hour >= 22 or event_dt.hour <= 5)
    row["hour_sin"] = math.sin(2 * math.pi * event_dt.hour / 24)
    row["hour_cos"] = math.cos(2 * math.pi * event_dt.hour / 24)
    row["day_sin"] = math.sin(2 * math.pi * event_dt.weekday() / 7)
    row["day_cos"] = math.cos(2 * math.pi * event_dt.weekday() / 7)
    report_lag = max(0.0, (event_dt - created_dt).total_seconds() / 60)
    row["report_lag_minutes_clipped"] = min(report_lag, 24 * 60)
    row["report_lag_hours_clipped"] = min(report_lag / 60, 24)
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
    row["location_grid"] = f"{row['lat_round_3']:.3f}_{row['lon_round_3']:.3f}"
    row["peak_period"] = peak_period
    row["start_day_name"] = event_dt.strftime("%A").lower()
    row["start_month_name"] = event_dt.strftime("%B").lower()
    row["event_cause_corridor"] = f"{event['event_cause']}_{event['corridor']}"
    row["cause_peak_interaction"] = f"{event['event_cause']}_{peak_period}"
    row["zone_cause_interaction"] = f"{event['zone']}_{event['event_cause']}"
    row["corridor_cause_interaction"] = f"{event['corridor']}_{event['event_cause']}"
    row["cause_heavy_vehicle_interaction"] = f"{event['event_cause']}_heavy_{row['is_heavy_vehicle']}"
    row["corridor_peak_interaction"] = f"{event['corridor']}_{peak_period}"
    row["cause_hour_interaction"] = f"{event['event_cause']}_{event_dt.hour}"
    row["vehicle_cause_interaction"] = f"{event['veh_type']}_{event['event_cause']}"
    row["cargo_vehicle_interaction"] = f"{event['veh_type']}_{event['veh_type']}"
    row["planned_peak_interaction"] = f"{event['event_type']}_{peak_period}"
    row["weather_zone_interaction"] = f"weather_{row['is_weather_or_visibility_event']}_{event['zone']}"
    row["road_closure_probability"] = closure_probability
    row["road_closure_probability_is_history_fallback"] = 0
    return row


def predict_duration_with_model2(bundle: dict, feature_df: pd.DataFrame, event: dict, closure_probability: float) -> dict:
    feature_cols = bundle["feature_cols"]
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    row = build_model2_feature_row(feature_df, event, closure_probability)
    event_x = row.to_frame().T.reindex(columns=feature_cols)
    for col in event_x.columns:
        event_x[col] = pd.to_numeric(event_x[col], errors="coerce")
    event_x = event_x.replace([float("inf"), float("-inf")], pd.NA)
    medians = feature_df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    event_x = event_x.fillna(medians).fillna(0)
    proba = model.predict_proba(event_x)[0]
    pred = model.predict(event_x)
    labels = list(label_encoder.classes_)
    duration = str(label_encoder.inverse_transform(pred)[0])
    return {
        "duration": duration,
        "duration_probs": {label: float(prob) for label, prob in zip(labels, proba)},
        "prediction_confidence": float(max(proba)),
        "model_source": "live_model2_predict_proba",
    }


def flatten_recommendation(event: dict, recommendation: dict) -> dict:
    row = {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        **event,
        "risk_score": round(float(recommendation["risk_score"]), 4),
        "risk_level": recommendation["risk_level"],
        "closure_probability": round(float(recommendation["closure_probability"]), 6),
        "predicted_duration": recommendation["duration"],
        "manpower": recommendation["manpower"],
        "barricading": recommendation["barricading"],
        "diversion": recommendation["diversion"],
        "alerts": recommendation["alerts"],
    }
    for key, value in recommendation["duration_probs"].items():
        row[f"prob_{key}"] = round(float(value), 6)
    for key, value in recommendation["text_flags"].items():
        row[key] = bool(value)
    return row


def append_event_log(event: dict, recommendation: dict) -> Path:
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = flatten_recommendation(event, recommendation)
    log_df = pd.DataFrame([row])
    should_write_header = not EVENT_LOG_PATH.exists()
    log_df.to_csv(EVENT_LOG_PATH, mode="a", index=False, header=should_write_header)
    return EVENT_LOG_PATH


@st.cache_data
def load_event_log() -> pd.DataFrame:
    if not EVENT_LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(EVENT_LOG_PATH)


def session_log_df() -> pd.DataFrame:
    rows = st.session_state.get("session_event_log", [])
    return pd.DataFrame(rows)


def build_recommendation(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model2_bundle: dict,
    event: dict,
) -> tuple[dict, pd.DataFrame]:
    candidates = df.copy()
    candidates["distance_km"] = haversine_km(
        event["latitude"],
        event["longitude"],
        candidates["latitude"],
        candidates["longitude"],
    )
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
        candidates["match_score"] += (candidates[column].astype(str) == str(event[column])).astype(float) * weight

    event_dt = parse_event_datetime(event["start_datetime"])
    if event_dt is not None and "start_hour" in candidates.columns:
        hour_gap = (candidates["start_hour"].fillna(event_dt.hour) - event_dt.hour).abs()
        candidates["match_score"] += (1 - hour_gap.clip(0, 12) / 12) * 0.6
    if event_dt is not None and "start_dayofweek" in candidates.columns:
        candidates["match_score"] += (candidates["start_dayofweek"].fillna(event_dt.weekday()) == event_dt.weekday()).astype(float) * 0.4

    candidates["similarity_score"] = candidates["match_score"] - candidates["distance_km"].clip(0, 20) / 8
    nearest = candidates.sort_values(["similarity_score", "distance_km"], ascending=[False, True]).head(50)

    closure_probability = float(nearest["road_closure_probability"].median())
    flags = text_risk_flags(event["description"])
    closure_probability += 0.08 if flags["has_blocked_word"] else 0
    closure_probability += 0.05 if flags["has_jam_word"] else 0
    closure_probability += 0.06 if flags["has_public_event_word"] else 0
    closure_probability += 0.04 if event["event_type"] == "planned" else 0
    closure_probability = max(0.01, min(0.98, closure_probability))

    model2_prediction = predict_duration_with_model2(model2_bundle, feature_df, event, closure_probability)
    duration_probs = model2_prediction["duration_probs"]
    duration = model2_prediction["duration"]

    duration_weight = {"short": 15, "medium": 35, "long": 60, "very_long": 80}.get(duration, 35)
    description_weight = (
        flags["has_blocked_word"] * 8
        + flags["has_jam_word"] * 5
        + flags["has_severity_word"] * 6
        + flags["has_diversion_word"] * 4
        + flags["has_public_event_word"] * 6
    )
    peak_weight = 5 if event_dt is not None and event_dt.hour in {8, 9, 10, 17, 18, 19, 20} else 0
    risk_score = min(100.0, closure_probability * 48 + duration_weight * 0.48 + description_weight + peak_weight)
    level = risk_level(risk_score)
    playbook = recommend_manpower(level, duration, closure_probability)

    return (
        {
            "risk_score": risk_score,
            "risk_level": level,
            "closure_probability": closure_probability,
            "duration": duration,
            "duration_probs": duration_probs,
            "prediction_confidence": model2_prediction["prediction_confidence"],
            "duration_model_source": model2_prediction["model_source"],
            "text_flags": flags,
            **playbook,
        },
        nearest,
    )


def event_input_form(df: pd.DataFrame) -> dict:
    st.subheader("New Event Input")
    st.caption("Enter the same fields available in the event feed. The app compares the scenario with similar historical events and generates a response playbook.")

    with st.form("event_input"):
        c1, c2, c3 = st.columns(3)
        with c1:
            event_type = st.selectbox("event_type", sorted(df["event_type"].unique()), index=0)
            event_cause = st.selectbox("Event cause", sorted(df["event_cause"].unique()), index=0)
            start_date = st.date_input("start_datetime date", value=datetime.now().date())
            start_clock = st.time_input("start_datetime time", value=time(18, 0))
        with c2:
            created_date = st.date_input("created_date", value=datetime.now().date())
            latitude = st.number_input("latitude", value=float(df["latitude"].median()), format="%.6f")
            longitude = st.number_input("longitude", value=float(df["longitude"].median()), format="%.6f")
        with c3:
            corridor = st.selectbox("Corridor", sorted(df["corridor"].unique()), index=0)
            police_station = st.selectbox("Police station", sorted(df["police_station"].unique()), index=0)
            junction = st.selectbox("junction", sorted(df["junction"].unique()), index=0)

        c4, c5 = st.columns(2)
        with c4:
            zone = st.selectbox("zone", sorted(df["zone"].unique()), index=0)
        with c5:
            veh_type = st.selectbox("veh_type", sorted(df["veh_type"].unique()), index=0)

        description = st.text_area("description", value="Political gathering near major junction, slow traffic expected.")
        submitted = st.form_submit_button("Generate Recommendation", type="primary")

    start_datetime = datetime.combine(start_date, start_clock).isoformat(timespec="minutes")
    return {
        "submitted": submitted,
        "event_type": event_type,
        "event_cause": event_cause,
        "start_datetime": start_datetime,
        "created_date": created_date.isoformat(),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "corridor": corridor,
        "police_station": police_station,
        "junction": junction,
        "zone": zone,
        "veh_type": veh_type,
        "description": description,
    }


def show_mappls_map(points: list[dict], center: tuple[float, float], token: str | None) -> None:
    if not token:
        st.map(pd.DataFrame(points).rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
        return

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta name="viewport" content="initial-scale=1.0, maximum-scale=1.0">
        <style>
          html, body, #map {{ margin:0; padding:0; height:520px; width:100%; }}
          .legend {{ position:absolute; left:14px; bottom:14px; background:white; padding:10px 12px; border-radius:8px; font:13px Arial; z-index:999; box-shadow:0 1px 8px rgba(15,23,42,.25); }}
          .dot {{ display:inline-block; width:10px; height:10px; border-radius:999px; margin-right:6px; }}
        </style>
      </head>
      <body>
        <div id="map"></div>
        <div class="legend">
          <div><span class="dot" style="background:{RISK_COLORS['critical']}"></span>Critical</div>
          <div><span class="dot" style="background:{RISK_COLORS['high']}"></span>High</div>
          <div><span class="dot" style="background:{RISK_COLORS['medium']}"></span>Medium</div>
          <div><span class="dot" style="background:{RISK_COLORS['low']}"></span>Low</div>
        </div>
        <script>
          const points = {json.dumps(points)};
          window.initMap = function() {{
            const map = new mappls.Map('map', {{
              center: [{center[0]}, {center[1]}],
              zoom: 11,
              zoomControl: true,
              geolocation: false
            }});
            map.addListener('load', function() {{
              points.forEach(function(p) {{
                const color = p.color || '#2563eb';
                const size = p.is_input ? 24 : 14;
                const marker = new mappls.Marker({{
                  map: map,
                  position: {{lat: p.latitude, lng: p.longitude}},
                  icon: {{
                    html: '<div style="width:' + size + 'px;height:' + size + 'px;border-radius:999px;background:' + color + ';border:3px solid white;box-shadow:0 0 0 2px rgba(15,23,42,.2);"></div>',
                    width: size,
                    height: size
                  }}
                }});
                marker.addListener('click', function() {{
                  new mappls.InfoWindow({{
                    map: map,
                    position: {{lat: p.latitude, lng: p.longitude}},
                    content: '<b>' + p.title + '</b><br>' + p.subtitle
                  }});
                }});
              }});
            }});
          }};
          const sdk = document.createElement('script');
          sdk.src = 'https://apis.mappls.com/advancedmaps/api/{token}/map_sdk?layer=vector&v=3.0&callback=initMap';
          document.head.appendChild(sdk);
        </script>
      </body>
    </html>
    """
    components.html(html, height=540, scrolling=False)


def build_heatmap_points(df: pd.DataFrame, max_points: int) -> list[dict]:
    heat_df = df.dropna(subset=["latitude", "longitude"]).copy()
    heat_df = heat_df[
        heat_df["latitude"].between(12.0, 14.0)
        & heat_df["longitude"].between(76.0, 78.5)
    ]
    if "road_closure_probability" in heat_df.columns:
        heat_df["heat_weight"] = heat_df["road_closure_probability"].clip(0.05, 1.0)
    else:
        heat_df["heat_weight"] = 0.5
    if "prob_very_long" in heat_df.columns:
        heat_df["heat_weight"] += heat_df["prob_very_long"].fillna(0) * 0.7
    if "prob_long" in heat_df.columns:
        heat_df["heat_weight"] += heat_df["prob_long"].fillna(0) * 0.4
    heat_df["heat_weight"] = heat_df["heat_weight"].clip(0.05, 1.8)
    heat_df = heat_df.sort_values("heat_weight", ascending=False).head(max_points)
    return [
        {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "weight": float(row["heat_weight"]),
            "title": str(row.get("id", "Event")),
            "subtitle": (
                f"{row.get('event_cause', 'unknown')} | "
                f"closure {float(row.get('road_closure_probability', 0)) * 100:.1f}%"
            ),
        }
        for _, row in heat_df.iterrows()
    ]


def show_mappls_heatmap(points: list[dict], center: tuple[float, float], token: str | None) -> None:
    if not token:
        st.warning("Mappls token unavailable. Showing Streamlit fallback map.")
        st.map(pd.DataFrame(points).rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
        return

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta name="viewport" content="initial-scale=1.0, maximum-scale=1.0">
        <style>
          html, body, #map-wrap, #map {{ margin:0; padding:0; height:560px; width:100%; }}
          #map-wrap {{ position:relative; background:#e5e7eb; }}
          #heat-canvas {{ position:absolute; inset:0; z-index:5; pointer-events:none; mix-blend-mode:multiply; }}
          .legend {{ position:absolute; left:14px; bottom:14px; background:white; padding:10px 12px; border-radius:8px; font:13px Arial; z-index:20; box-shadow:0 1px 8px rgba(15,23,42,.25); }}
          .gradient {{ width:150px; height:12px; border-radius:999px; background:linear-gradient(90deg, rgba(34,197,94,.1), #fde047, #f97316, #dc2626); margin:7px 0; }}
          .note {{ position:absolute; top:14px; right:14px; background:white; padding:8px 10px; border-radius:8px; font:12px Arial; z-index:20; box-shadow:0 1px 8px rgba(15,23,42,.2); }}
        </style>
      </head>
      <body>
        <div id="map-wrap">
          <div id="map"></div>
          <canvas id="heat-canvas"></canvas>
          <div class="note">Risk density = closure probability + long-duration likelihood</div>
          <div class="legend">
            <strong>Heat Intensity</strong>
            <div class="gradient"></div>
            <div style="display:flex; justify-content:space-between; gap:16px;"><span>Lower</span><span>Higher</span></div>
          </div>
        </div>
        <script>
          const points = {json.dumps(points)};
          const center = [{center[0]}, {center[1]}];
          const zoom = 11;

          function mercatorProject(lat, lon, zoomLevel, width, height) {{
            const tileSize = 256;
            const scale = tileSize * Math.pow(2, zoomLevel);
            const lngX = (lon + 180) / 360 * scale;
            const sinLat = Math.sin(lat * Math.PI / 180);
            const latY = (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale;
            const centerX = (center[1] + 180) / 360 * scale;
            const centerSinLat = Math.sin(center[0] * Math.PI / 180);
            const centerY = (0.5 - Math.log((1 + centerSinLat) / (1 - centerSinLat)) / (4 * Math.PI)) * scale;
            return {{
              x: width / 2 + (lngX - centerX),
              y: height / 2 + (latY - centerY)
            }};
          }}

          function drawHeatmap() {{
            const canvas = document.getElementById('heat-canvas');
            const wrap = document.getElementById('map-wrap');
            const ratio = window.devicePixelRatio || 1;
            const width = wrap.clientWidth;
            const height = wrap.clientHeight;
            canvas.width = width * ratio;
            canvas.height = height * ratio;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            const ctx = canvas.getContext('2d');
            ctx.scale(ratio, ratio);
            ctx.clearRect(0, 0, width, height);

            points.forEach(function(p) {{
              const px = mercatorProject(p.latitude, p.longitude, zoom, width, height);
              const radius = 24 + Math.min(44, p.weight * 28);
              const alpha = Math.min(0.42, 0.12 + p.weight * 0.14);
              const gradient = ctx.createRadialGradient(px.x, px.y, 0, px.x, px.y, radius);
              gradient.addColorStop(0, 'rgba(220,38,38,' + alpha + ')');
              gradient.addColorStop(0.35, 'rgba(249,115,22,' + (alpha * 0.7) + ')');
              gradient.addColorStop(0.68, 'rgba(253,224,71,' + (alpha * 0.34) + ')');
              gradient.addColorStop(1, 'rgba(34,197,94,0)');
              ctx.fillStyle = gradient;
              ctx.beginPath();
              ctx.arc(px.x, px.y, radius, 0, Math.PI * 2);
              ctx.fill();
            }});
          }}

          window.initMap = function() {{
            const map = new mappls.Map('map', {{
              center: center,
              zoom: zoom,
              zoomControl: true,
              geolocation: false
            }});
            map.addListener('load', function() {{
              drawHeatmap();
              points.slice(0, 60).forEach(function(p) {{
                const marker = new mappls.Marker({{
                  map: map,
                  position: {{lat: p.latitude, lng: p.longitude}},
                  icon: {{
                    html: '<div style="width:7px;height:7px;border-radius:999px;background:#111827;border:1px solid white;opacity:.75"></div>',
                    width: 7,
                    height: 7
                  }}
                }});
                marker.addListener('click', function() {{
                  new mappls.InfoWindow({{
                    map: map,
                    position: {{lat: p.latitude, lng: p.longitude}},
                    content: '<b>' + p.title + '</b><br>' + p.subtitle
                  }});
                }});
              }});
            }});
            window.addEventListener('resize', drawHeatmap);
          }};

          const sdk = document.createElement('script');
          sdk.src = 'https://apis.mappls.com/advancedmaps/api/{token}/map_sdk?layer=vector&v=3.0&callback=initMap';
          document.head.appendChild(sdk);
        </script>
      </body>
    </html>
    """
    components.html(html, height=580, scrolling=False)


def main() -> None:
    st.title("EventOps AI Command Board")
    st.caption("Forecast event traffic impact and recommend manpower, barricading, and diversion response.")

    if not HANDOFF_PATH.exists():
        st.error(f"Missing input file: {HANDOFF_PATH}")
        st.stop()

    df = load_handoff_data()
    feature_df = load_model2_feature_data()
    model2_bundle = load_model2_bundle()
    client_id = get_secret("CLIENT_ID")
    client_secret = get_secret("CLIENT_SECRET")
    token_result = get_mappls_access_token(client_id, client_secret) if client_id and client_secret else {"ok": False}
    access_token = token_result.get("access_token") if token_result.get("ok") else None

    with st.sidebar:
        st.header("Mappls")
        if access_token:
            st.success("OAuth token ready")
        else:
            st.warning("Mappls token unavailable")
            if token_result.get("error"):
                st.caption(token_result["error"])

    tab_input, tab_map, tab_heatmap, tab_data = st.tabs(["Input + Recommendation", "Mappls Map", "Risk Heatmap", "Historical Data"])

    with tab_input:
        event = event_input_form(df)
        if event["submitted"]:
            recommendation, nearest = build_recommendation(df, feature_df, model2_bundle, event)
            st.session_state["latest_event"] = event
            st.session_state["latest_recommendation"] = recommendation
            st.session_state["latest_nearest"] = nearest
            st.session_state.setdefault("session_event_log", []).append(flatten_recommendation(event, recommendation))

        if "latest_recommendation" in st.session_state:
            recommendation = st.session_state["latest_recommendation"]
            nearest = st.session_state["latest_nearest"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risk Level", recommendation["risk_level"].upper())
            c2.metric("Risk Score", f"{recommendation['risk_score']:.1f}/100")
            c3.metric("Closure Probability", f"{recommendation['closure_probability'] * 100:.1f}%")
            c4.metric("Predicted Duration", recommendation["duration"].replace("_", " ").title())
            st.caption(f"Duration source: `{recommendation['duration_model_source']}` | confidence: `{recommendation['prediction_confidence']:.3f}`")

            left, right = st.columns([1, 1])
            with left:
                st.subheader("Recommended Deployment")
                st.write(f"**Manpower:** {recommendation['manpower']}")
                st.write(f"**Barricading:** {recommendation['barricading']}")
                st.write(f"**Diversion:** {recommendation['diversion']}")
                st.write(f"**Alerts:** {recommendation['alerts']}")
                if st.button("Save Event + Recommendation", type="secondary"):
                    saved_path = append_event_log(st.session_state["latest_event"], recommendation)
                    load_event_log.clear()
                    st.success(f"Saved locally to `{saved_path.relative_to(PROJECT_ROOT)}`")
                    st.caption("For Vercel-style deployments, use the download button too because local filesystem writes are not durable.")
            with right:
                st.subheader("Duration Probability")
                probs = pd.DataFrame(
                    [{"duration_band": key, "probability": value} for key, value in recommendation["duration_probs"].items()]
                )
                st.bar_chart(probs, x="duration_band", y="probability", color="#2563eb")
                st.write("**Text signals used**")
                st.json(recommendation["text_flags"])

            st.subheader("Similar Historical Events")
            cols = [
                "id",
                "start_datetime",
                "event_cause",
                "event_type",
                "corridor",
                "police_station",
                "junction",
                "zone",
                "veh_type",
                "duration_band",
                "predicted_duration_band",
                "road_closure_probability",
                "distance_km",
                "similarity_score",
            ]
            st.dataframe(nearest[cols].head(10), use_container_width=True, hide_index=True)

            current_row = pd.DataFrame([flatten_recommendation(st.session_state["latest_event"], recommendation)])
            st.download_button(
                "Download This Recommendation CSV",
                data=current_row.to_csv(index=False).encode("utf-8"),
                file_name="eventops_recommendation.csv",
                mime="text/csv",
            )
        else:
            st.info("Fill the event fields and click Generate Recommendation.")

    with tab_map:
        st.subheader("Mappls Event View")
        latest_event = st.session_state.get("latest_event")
        latest_recommendation = st.session_state.get("latest_recommendation")
        if latest_event and latest_recommendation:
            level = latest_recommendation["risk_level"]
            points = [
                {
                    "latitude": latest_event["latitude"],
                    "longitude": latest_event["longitude"],
                    "title": "New input event",
                    "subtitle": f"{level.upper()} risk | {latest_recommendation['duration']}",
                    "color": RISK_COLORS[level],
                    "is_input": True,
                }
            ]
            nearest = st.session_state["latest_nearest"].head(30)
            for _, row in nearest.iterrows():
                points.append(
                    {
                        "latitude": float(row["latitude"]),
                        "longitude": float(row["longitude"]),
                        "title": str(row.get("id", "Historical event")),
                        "subtitle": f"{row.get('event_cause', 'unknown')} | {row.get('duration_band', 'unknown')}",
                        "color": "#64748b",
                        "is_input": False,
                    }
                )
            show_mappls_map(points, (latest_event["latitude"], latest_event["longitude"]), access_token)
        else:
            sample = df.sort_values("road_closure_probability", ascending=False).head(100)
            points = [
                {
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "title": str(row.get("id", "Historical event")),
                    "subtitle": f"{row.get('event_cause', 'unknown')} | closure {float(row.get('road_closure_probability', 0)) * 100:.1f}%",
                    "color": "#2563eb",
                    "is_input": False,
                }
                for _, row in sample.iterrows()
            ]
            show_mappls_map(points, (float(sample["latitude"].mean()), float(sample["longitude"].mean())), access_token)

    with tab_heatmap:
        st.subheader("Mappls Risk Heatmap")
        st.caption("This shows operational hotspot density across historical events. Darker/warmer areas mean higher closure probability and longer-duration likelihood.")
        c1, c2, c3 = st.columns(3)
        with c1:
            max_points = st.slider("Heatmap events", 100, 2000, 800, step=100)
        with c2:
            selected_causes = st.multiselect("Cause filter", sorted(df["event_cause"].unique()))
        with c3:
            selected_corridors = st.multiselect("Corridor filter", sorted(df["corridor"].unique()))

        heat_df = df.copy()
        if selected_causes:
            heat_df = heat_df[heat_df["event_cause"].isin(selected_causes)]
        if selected_corridors:
            heat_df = heat_df[heat_df["corridor"].isin(selected_corridors)]

        heat_points = build_heatmap_points(heat_df, max_points)
        if heat_points:
            center = (
                sum(point["latitude"] for point in heat_points) / len(heat_points),
                sum(point["longitude"] for point in heat_points) / len(heat_points),
            )
            show_mappls_heatmap(heat_points, center, access_token)

            heat_table = pd.DataFrame(heat_points)
            st.write("**Top heat contributors**")
            st.dataframe(
                heat_table.sort_values("weight", ascending=False).head(20),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No heatmap points match the current filters.")

    with tab_data:
        st.subheader("Model Handoff Data")
        st.write(f"Rows: {len(df):,}")
        st.dataframe(df.head(200), use_container_width=True, hide_index=True)

        st.subheader("Saved Recommendation Log")
        local_log = load_event_log()
        active_log = local_log if not local_log.empty else session_log_df()
        if active_log.empty:
            st.info("No saved recommendations yet. Generate a recommendation and click Save Event + Recommendation.")
        else:
            st.write(f"Saved rows: {len(active_log):,}")
            st.dataframe(active_log.sort_values("logged_at", ascending=False), use_container_width=True, hide_index=True)
            st.download_button(
                "Download Recommendation Log CSV",
                data=active_log.to_csv(index=False).encode("utf-8"),
                file_name="event_recommendation_log.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()

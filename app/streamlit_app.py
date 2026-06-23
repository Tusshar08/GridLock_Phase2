from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pipeline_engine import build_heatmap_points, run_full_pipeline, run_live_event_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RISK_COLORS = {"low": "#0f766e", "medium": "#ca8a04", "high": "#ea580c", "critical": "#dc2626"}

st.set_page_config(page_title="RoadGuard AI", page_icon="🚦", layout="wide")


@st.cache_resource
def load_runtime_artifacts():
    return run_full_pipeline(PROJECT_ROOT, persist_outputs=True)


@st.cache_data
def load_input_reference_data() -> pd.DataFrame:
    columns = [
        "event_type",
        "event_cause",
        "corridor",
        "police_station",
        "junction",
        "zone",
        "veh_type",
        "latitude",
        "longitude",
    ]
    candidates = [
        PROJECT_ROOT / "outputs" / "features" / "road_closure_features_v1.csv",
        PROJECT_ROOT / "outputs" / "features" / "duration_base_features_v1.csv",
        PROJECT_ROOT / "outputs" / "model_road_closure" / "model2_duration_handoff.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, usecols=lambda col: col in columns, low_memory=False)
    return pd.DataFrame(columns=columns)


def clean_options(values: pd.Series) -> list[str]:
    options = values.dropna().astype(str).str.strip()
    options = options[options.ne("") & options.str.lower().ne("nan")]
    unique_options = sorted(options.unique())
    return unique_options if unique_options else ["unknown"]


def display_option(value: str) -> str:
    return str(value).replace("_", " ").title()


def get_secret(name: str) -> str:
    try:
        value = str(st.secrets.get(name, "")).strip()
        if value:
            return value
    except Exception:
        pass
    return str(os.environ.get(name, "")).strip()


@st.cache_data(ttl=3300)
def fetch_mappls_oauth_token(client_id: str, client_secret: str) -> dict:
    data = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://outpost.mappls.com/api/security/oauth/token",
        data=data,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_mappls_auth() -> dict:
    client_id = get_secret("CLIENT_ID")
    client_secret = get_secret("CLIENT_SECRET")
    if client_id and client_secret:
        try:
            token_result = fetch_mappls_oauth_token(client_id, client_secret)
            access_token = str(token_result.get("access_token", "")).strip()
            if access_token:
                return {"ok": True, "access_token": access_token, "source": "oauth_client_credentials"}
        except Exception as exc:
            return {"ok": False, "source": "oauth_error", "error": f"Mappls OAuth token generation failed: {exc}"}

    static_key = get_secret("MAPPLS_STATIC_KEY") or get_secret("MAPPLS_ACCESS_TOKEN")
    if static_key:
        return {"ok": True, "access_token": static_key, "source": "secrets_static_key"}

    return {
        "ok": False,
        "source": "missing_secret",
        "error": "Missing MAPPLS_STATIC_KEY in .streamlit/secrets.toml.",
    }


def event_input_form(history_df: pd.DataFrame, input_reference_df: pd.DataFrame | None = None) -> dict:
    st.subheader("Report a New Traffic Event")
    st.caption("Enter what is known now. The app will estimate risk and suggest the first response plan.")

    input_df = input_reference_df if input_reference_df is not None and not input_reference_df.empty else history_df
    selector_df = pd.concat([input_df, history_df], ignore_index=True, sort=False)
    for col in ["corridor", "police_station", "junction", "zone"]:
        if col not in selector_df.columns:
            selector_df[col] = "unknown"

    with st.form("event_input"):
        c1, c2, c3 = st.columns(3)
        with c1:
            event_type = st.selectbox("Event type", clean_options(selector_df["event_type"]), index=0, format_func=display_option)
            event_cause = st.selectbox("Main cause", clean_options(selector_df["event_cause"]), index=0, format_func=display_option)
            start_date = st.date_input("Event date", value=datetime.now().date())
            start_clock = st.time_input("Event time", value=time(18, 0))
        with c2:
            created_date = st.date_input("Reported date", value=datetime.now().date())
            latitude = st.number_input("Latitude", value=float(pd.to_numeric(selector_df["latitude"], errors="coerce").median()), format="%.6f")
            longitude = st.number_input("Longitude", value=float(pd.to_numeric(selector_df["longitude"], errors="coerce").median()), format="%.6f")
        with c3:
            corridor = st.selectbox("Road corridor", clean_options(selector_df["corridor"]), index=0, format_func=display_option)
            police_station = st.selectbox("Police station", clean_options(selector_df["police_station"]), index=0, format_func=display_option)
            junction = st.selectbox("Junction", clean_options(selector_df["junction"]), index=0, format_func=display_option)

        c4, c5 = st.columns(2)
        with c4:
            zone = st.selectbox("City zone", clean_options(selector_df["zone"]), index=0, format_func=display_option)
        with c5:
            veh_type = st.selectbox("Vehicle type", clean_options(selector_df["veh_type"]), index=0, format_func=display_option)

        description = st.text_area("What happened?", value="Traffic obstruction reported near major junction, congestion expected.")
        submitted = st.form_submit_button("Get Response Plan", type="primary")

    return {
        "submitted": submitted,
        "event_type": event_type,
        "event_cause": event_cause,
        "start_datetime": datetime.combine(start_date, start_clock).isoformat(timespec="minutes"),
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
              scrollZoom: true,
              doubleClickZoom: true,
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
          <div class="note">Darker areas may need faster attention</div>
          <div class="legend">
            <strong>Heat Intensity</strong>
            <div class="gradient"></div>
            <div style="display:flex; justify-content:space-between; gap:16px;"><span>Lower</span><span>Higher</span></div>
          </div>
        </div>
        <script>
          const points = {json.dumps(points)};
          const initialCenter = [{center[0]}, {center[1]}];
          const zoom = 11;

          function getMapCenter(map) {{
            try {{
              const current = map && map.getCenter ? map.getCenter() : null;
              if (Array.isArray(current) && current.length >= 2) {{
                return [Number(current[0]), Number(current[1])];
              }}
              if (current) {{
                const lat = Number(current.lat ?? current.latitude);
                const lng = Number(current.lng ?? current.lon ?? current.longitude);
                if (Number.isFinite(lat) && Number.isFinite(lng)) {{
                  return [lat, lng];
                }}
              }}
            }} catch (error) {{}}
            return initialCenter;
          }}

          function getMapZoom(map) {{
            try {{
              const currentZoom = map && map.getZoom ? Number(map.getZoom()) : zoom;
              return Number.isFinite(currentZoom) ? currentZoom : zoom;
            }} catch (error) {{
              return zoom;
            }}
          }}

          function mercatorProject(lat, lon, zoomLevel, width, height, mapCenter) {{
            const tileSize = 256;
            const scale = tileSize * Math.pow(2, zoomLevel);
            const lngX = (lon + 180) / 360 * scale;
            const sinLat = Math.sin(lat * Math.PI / 180);
            const latY = (0.5 - Math.log((1 + sinLat) / (1 - sinLat)) / (4 * Math.PI)) * scale;
            const centerX = (mapCenter[1] + 180) / 360 * scale;
            const centerSinLat = Math.sin(mapCenter[0] * Math.PI / 180);
            const centerY = (0.5 - Math.log((1 + centerSinLat) / (1 - centerSinLat)) / (4 * Math.PI)) * scale;
            return {{
              x: width / 2 + (lngX - centerX),
              y: height / 2 + (latY - centerY)
            }};
          }}

          function drawHeatmap(map) {{
            const canvas = document.getElementById('heat-canvas');
            const wrap = document.getElementById('map-wrap');
            const ratio = window.devicePixelRatio || 1;
            const width = wrap.clientWidth;
            const height = wrap.clientHeight;
            const mapCenter = getMapCenter(map);
            const mapZoom = getMapZoom(map);
            canvas.width = width * ratio;
            canvas.height = height * ratio;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            const ctx = canvas.getContext('2d');
            ctx.scale(ratio, ratio);
            ctx.clearRect(0, 0, width, height);

                        points.forEach(function(p) {{
              const px = mercatorProject(p.latitude, p.longitude, mapZoom, width, height, mapCenter);
                            const radius = (p.is_live ? 42 : 24) + Math.min(44, p.weight * 28);
                            const alpha = Math.min(0.5, (p.is_live ? 0.22 : 0.12) + p.weight * 0.14);
              const gradient = ctx.createRadialGradient(px.x, px.y, 0, px.x, px.y, radius);
                            gradient.addColorStop(0, p.is_live ? 'rgba(37,99,235,' + alpha + ')' : 'rgba(220,38,38,' + alpha + ')');
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
              center: initialCenter,
              zoom: zoom,
              zoomControl: true,
              scrollZoom: true,
              doubleClickZoom: true,
              geolocation: false
            }});
            let pendingDraw = null;
            function scheduleHeatmapDraw() {{
              if (pendingDraw) {{
                window.cancelAnimationFrame(pendingDraw);
              }}
              pendingDraw = window.requestAnimationFrame(function() {{
                drawHeatmap(map);
                pendingDraw = null;
              }});
            }}
            map.addListener('load', function() {{
              scheduleHeatmapDraw();
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
            ['move', 'drag', 'dragend', 'zoom', 'zoomend'].forEach(function(eventName) {{
              try {{
                map.addListener(eventName, scheduleHeatmapDraw);
              }} catch (error) {{}}
            }});
            window.addEventListener('resize', scheduleHeatmapDraw);
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
    st.title("RoadGuard AI")
    st.caption("Quick risk estimate, response guidance, and hotspot view for traffic events.")

    artifacts = load_runtime_artifacts()
    df = artifacts.risk_playbooks
    input_reference_df = load_input_reference_data()

    token_result = resolve_mappls_auth()
    access_token = token_result.get("access_token") if token_result.get("ok") else None

    tab_input, tab_heatmap = st.tabs(["Response Plan", "Hotspot View"])

    with tab_input:
        event = event_input_form(df, input_reference_df)
        if event["submitted"]:
            result = run_live_event_workflow(PROJECT_ROOT, artifacts, event)
            st.session_state["latest_result"] = result

        if "latest_result" in st.session_state:
            result = st.session_state["latest_result"]
            rec = result.recommendation_row.iloc[0]
            closure_probability = float(result.model1_prediction.get("road_closure_probability", 0.0))
            predicted_band = str(result.model2_prediction.get("predicted_duration_band", "unknown")).replace("_", " ").title()
            prediction_confidence = float(result.model2_prediction.get("prediction_confidence", rec.get("prediction_confidence", 0.0)))
            if prediction_confidence >= 0.70:
              confidence_note = "High"
            elif prediction_confidence >= 0.50:
              confidence_note = "Moderate"
            else:
              confidence_note = "Low"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall Risk", str(rec["risk_level"]).upper())
            c2.metric("Risk Score", f"{float(rec['risk_score']):.1f}/100")
            c3.metric("Chance of Road Closure", f"{closure_probability * 100:.1f}%")
            c4.metric("Expected Clearance Time", predicted_band)

            st.subheader("Situation Snapshot")
            s1, s2, s3 = st.columns(3)
            s1.info(f"Reliability of this estimate: {prediction_confidence * 100:.1f}% ({confidence_note})")
            s2.info(f"Past activity in this area: {str(rec.get('hotspot_level', 'n/a')).replace('_', ' ').title()}")
            s3.info(f"Local pressure score: {float(rec.get('hotspot_score', 0.0)):.2f}")

            st.subheader("Recommended Field Plan")
            manpower = str(rec.get("manpower") or "2 officers + 1 patrol unit")
            barricading = str(rec.get("barricading") or "Local channelization with cones/signage near incident point.")
            diversion = str(rec.get("diversion") or "Prepare local diversion if queue spills to adjacent junction.")
            control_room = str(rec.get("control_room") or "Notify station control and monitor every 15 minutes.")
            equipment = str(rec.get("equipment") or "reflective jackets, traffic cones")

            field_plan = pd.DataFrame(
                [
                    {"Area": "People to send", "Recommendation": manpower},
                    {"Area": "Barricading", "Recommendation": barricading},
                    {"Area": "Diversion plan", "Recommendation": diversion},
                    {"Area": "Control room update", "Recommendation": control_room},
                    {"Area": "Equipment to carry", "Recommendation": equipment},
                ]
            )
            st.dataframe(field_plan, use_container_width=True, hide_index=True)

            st.subheader("Priority Actions")
            action_items = []
            if closure_probability >= 0.50:
                action_items.append("Keep one diversion route ready before traffic backs up.")
            else:
                action_items.append("Monitor the approach roads and prepare diversion only if queues start building.")
            if str(rec.get("risk_level", "")).lower() in {"high", "critical"}:
                action_items.append("Send the first response team immediately and update the control room after arrival.")
            else:
                action_items.append("Assign a patrol unit and reassess once the field team confirms the scene.")
            if predicted_band.lower() in {"long", "very long", "very_long"}:
                action_items.append("Plan for a longer clearance window and alert nearby junction teams.")
            else:
                action_items.append("Expect a shorter clearance window, but keep equipment ready in case the scene escalates.")
            if float(rec.get("hotspot_score", 0.0)) >= 50:
                action_items.append("Treat this as a known pressure point and watch spillover at nearby junctions.")
            action_cols = st.columns(len(action_items))
            for index, (column, item) in enumerate(zip(action_cols, action_items), start=1):
                with column:
                    st.info(f"Step {index}\n\n{item}")

            st.subheader("Similar Past Events")
            similar_cols = {
                "start_datetime": "When",
                "event_cause": "Cause",
                "corridor": "Corridor",
                "police_station": "Police Station",
                "junction": "Junction",
                "zone": "Zone",
                "predicted_duration_band": "Clearance Time",
                "risk_level": "Risk",
                "risk_score": "Score",
            }
            cols = [c for c in similar_cols if c in result.similar_incidents.columns]
            st.dataframe(
                result.similar_incidents[cols].head(10).rename(columns=similar_cols),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Response Details",
                data=result.recommendation_row.to_csv(index=False).encode("utf-8"),
                file_name="roadguard_ai_response_details.csv",
                mime="text/csv",
            )
        else:
            st.info("Fill the event details and click Get Response Plan.")

    with tab_heatmap:
        st.subheader("City Hotspots")
        st.caption("Use this view to see where events may need extra attention. Your latest reported event is highlighted when available.")
        c1, c2, c3 = st.columns(3)
        with c1:
            max_points = st.slider("Events to show", 100, 2000, 800, step=100)
        with c2:
            selected_causes = st.multiselect("Show only these causes", sorted(df["event_cause"].unique()), format_func=display_option)
        with c3:
            selected_corridors = st.multiselect("Show only these corridors", sorted(df["corridor"].unique()), format_func=display_option)

        live_event_row = None
        latest_result = st.session_state.get("latest_result")
        if latest_result is not None:
            live_event_row = latest_result.recommendation_row.iloc[0]

        heat_points = build_heatmap_points(
            artifacts.risk_playbooks,
            max_points=max_points,
            selected_causes=selected_causes,
            selected_corridors=selected_corridors,
            live_event_row=live_event_row,
        )
        if heat_points:
            center = (
                sum(point["latitude"] for point in heat_points) / len(heat_points),
                sum(point["longitude"] for point in heat_points) / len(heat_points),
            )
            show_mappls_heatmap(heat_points, center, access_token)
        else:
            st.info("No heatmap points match the current filters.")

if __name__ == "__main__":
    main()

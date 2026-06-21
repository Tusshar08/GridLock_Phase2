from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pipeline_engine import build_heatmap_points, run_full_pipeline, run_live_event_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RISK_COLORS = {"low": "#0f766e", "medium": "#ca8a04", "high": "#ea580c", "critical": "#dc2626"}

st.set_page_config(page_title="EventOps AI", page_icon="🚦", layout="wide")


@st.cache_resource
def load_runtime_artifacts():
    return run_full_pipeline(PROJECT_ROOT, persist_outputs=True)


def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def resolve_mappls_auth() -> dict:
    # Runtime OAuth generation is intentionally disabled; use pre-provisioned secret only.
    static_key = get_secret("MAPPLS_STATIC_KEY") or get_secret("MAPPLS_ACCESS_TOKEN")
    if static_key:
        return {"ok": True, "access_token": static_key, "source": "secrets_static_key"}

    has_oauth_pair = bool(get_secret("CLIENT_ID") and get_secret("CLIENT_SECRET"))
    if has_oauth_pair:
        return {
            "ok": False,
            "source": "oauth_disabled",
            "error": "CLIENT_ID and CLIENT_SECRET are present, but runtime OAuth token generation is disabled. Add MAPPLS_STATIC_KEY to .streamlit/secrets.toml.",
        }

    return {
        "ok": False,
        "source": "missing_secret",
        "error": "Missing MAPPLS_STATIC_KEY in .streamlit/secrets.toml.",
    }


def event_input_form(history_df: pd.DataFrame) -> dict:
    st.subheader("New Event Input")
    st.caption("Input fields feed the complete chain: Model1 -> Model2 -> hotspot analytics -> similarity retrieval -> recommendation engine.")

    # Constrain location selectors to historically valid combinations so hotspot mapping is meaningful.
    location_df = (
        history_df[["corridor", "police_station", "junction"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    with st.form("event_input"):
        c1, c2, c3 = st.columns(3)
        with c1:
            event_type = st.selectbox("event_type", sorted(history_df["event_type"].dropna().astype(str).unique()), index=0)
            event_cause = st.selectbox("event_cause", sorted(history_df["event_cause"].dropna().astype(str).unique()), index=0)
            start_date = st.date_input("start_datetime (date)", value=datetime.now().date())
            start_clock = st.time_input("start_datetime (time)", value=time(18, 0))
        with c2:
            created_date = st.date_input("created_date", value=datetime.now().date())
            latitude = st.number_input("latitude", value=float(pd.to_numeric(history_df["latitude"], errors="coerce").median()), format="%.6f")
            longitude = st.number_input("longitude", value=float(pd.to_numeric(history_df["longitude"], errors="coerce").median()), format="%.6f")
        with c3:
            corridor_options = sorted(location_df["corridor"].unique()) if not location_df.empty else sorted(history_df["corridor"].dropna().astype(str).unique())
            corridor = st.selectbox("corridor", corridor_options, index=0)

            corridor_rows = location_df[location_df["corridor"] == corridor]
            police_options = (
                sorted(corridor_rows["police_station"].unique())
                if not corridor_rows.empty
                else sorted(history_df["police_station"].dropna().astype(str).unique())
            )
            police_station = st.selectbox("police_station", police_options, index=0)

            combo_rows = corridor_rows[corridor_rows["police_station"] == police_station]
            junction_options = (
                sorted(combo_rows["junction"].unique())
                if not combo_rows.empty
                else sorted(corridor_rows["junction"].unique()) if not corridor_rows.empty else sorted(history_df["junction"].dropna().astype(str).unique())
            )
            junction = st.selectbox("junction", junction_options, index=0)

        c4, c5 = st.columns(2)
        with c4:
            zone = st.selectbox("zone", sorted(history_df["zone"].dropna().astype(str).unique()), index=0)
        with c5:
            veh_type = st.selectbox("veh_type", sorted(history_df["veh_type"].dropna().astype(str).unique()), index=0)

        description = st.text_area("description", value="Traffic obstruction reported near major junction, congestion expected.")
        submitted = st.form_submit_button("Run Full Workflow", type="primary")

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
    st.caption("Dashboard-only UI. All model and analytics logic runs inside pipeline engine.")

    artifacts = load_runtime_artifacts()
    df = artifacts.risk_playbooks

    token_result = resolve_mappls_auth()
    access_token = token_result.get("access_token") if token_result.get("ok") else None

    with st.sidebar:
      st.header("Mappls")
      if access_token:
        st.success("Map key loaded from secrets.toml")
      else:
        st.warning("Mappls token unavailable")
        if token_result.get("error"):
          st.caption(token_result["error"])

    tab_input, tab_map, tab_heatmap, tab_data = st.tabs(["Input + Predictions", "Mappls Map", "Risk Heatmap", "Pipeline Data"])

    with tab_input:
        event = event_input_form(df)
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
            c1.metric("Risk Level", str(rec["risk_level"]).upper())
            c2.metric("Risk Score", f"{float(rec['risk_score']):.1f}/100")
            c3.metric("Road Closure Probability", f"{closure_probability * 100:.2f}%")
            c4.metric("Duration Band Prediction", predicted_band)

            st.subheader("Prediction Summary")
            s1, s2, s3 = st.columns(3)
            s1.info(f"Prediction Confidence: {prediction_confidence * 100:.1f}% ({confidence_note})")
            s2.info(f"Hotspot Level: {str(rec.get('hotspot_level', 'n/a')).replace('_', ' ').title()}")
            s3.info(f"Hotspot Score: {float(rec.get('hotspot_score', 0.0)):.2f}")

            st.subheader("Operational Response Plan")
            manpower = str(rec.get("manpower") or "2 officers + 1 patrol unit")
            barricading = str(rec.get("barricading") or "Local channelization with cones/signage near incident point.")
            diversion = str(rec.get("diversion") or "Prepare local diversion if queue spills to adjacent junction.")
            control_room = str(rec.get("control_room") or "Notify station control and monitor every 15 minutes.")
            equipment = str(rec.get("equipment") or "reflective jackets, traffic cones")

            o1, o2 = st.columns(2)
            with o1:
              st.markdown(f"**Manpower**\n\n{manpower}")
              st.markdown(f"**Barricading**\n\n{barricading}")
              st.markdown(f"**Diversion**\n\n{diversion}")
            with o2:
              st.markdown(f"**Control Room**\n\n{control_room}")
              st.markdown(f"**Equipment**\n\n{equipment}")

            probs = pd.DataFrame(
                [{"duration_band": key, "probability": value} for key, value in result.model2_prediction["duration_probs"].items()]
            )
            st.bar_chart(probs, x="duration_band", y="probability", color="#2563eb")

            st.subheader("Similar Historical Incidents")
            cols = [
                "id",
                "start_datetime",
                "event_type",
                "event_cause",
                "corridor",
                "police_station",
                "junction",
                "zone",
                "veh_type",
                "road_closure_probability",
                "predicted_duration_band",
                "risk_level",
                "risk_score",
                "similarity_score",
            ]
            cols = [c for c in cols if c in result.similar_incidents.columns]
            st.dataframe(result.similar_incidents[cols].head(10), use_container_width=True, hide_index=True)

            st.download_button(
                "Download Current Prediction CSV",
                data=result.recommendation_row.to_csv(index=False).encode("utf-8"),
                file_name="eventops_current_prediction.csv",
                mime="text/csv",
            )
        else:
            st.info("Fill the event fields and click Run Full Workflow.")

    with tab_map:
        st.subheader("Mappls Event View")
        latest_result = st.session_state.get("latest_result")
        if latest_result is not None:
            rec = latest_result.recommendation_row.iloc[0]
            event_row = latest_result.event_row.iloc[0]
            level = str(rec["risk_level"])
            points = [
                {
                    "latitude": float(event_row["latitude"]),
                    "longitude": float(event_row["longitude"]),
                    "title": "New input event",
                    "subtitle": f"{level.upper()} risk | {rec['predicted_duration_band']}",
                    "color": RISK_COLORS[level],
                    "is_input": True,
                }
            ]
            for _, row in latest_result.similar_incidents.head(30).iterrows():
                points.append(
                    {
                        "latitude": float(row.get("latitude", event_row["latitude"])),
                        "longitude": float(row.get("longitude", event_row["longitude"])),
                        "title": str(row.get("id", "Historical event")),
                        "subtitle": f"{row.get('event_cause', 'unknown')} | {row.get('predicted_duration_band', 'unknown')}",
                        "color": "#64748b",
                        "is_input": False,
                    }
                )
            show_mappls_map(points, (float(event_row["latitude"]), float(event_row["longitude"])), access_token)
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
        st.caption("Hotspot analytics from the pipeline are rendered on Mappls heatmap. Live event impact is overlaid in blue glow.")
        c1, c2, c3 = st.columns(3)
        with c1:
            max_points = st.slider("Heatmap events", 100, 2000, 800, step=100)
        with c2:
            selected_causes = st.multiselect("Cause filter", sorted(df["event_cause"].unique()))
        with c3:
            selected_corridors = st.multiselect("Corridor filter", sorted(df["corridor"].unique()))

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
        st.subheader("Pipeline Artifacts")
        st.write(f"Historical rows used: {len(df):,}")
        st.dataframe(df.head(200), use_container_width=True, hide_index=True)

        st.subheader("Recommendation Summary")
        st.dataframe(artifacts.recommendation_summary, use_container_width=True, hide_index=True)
        st.subheader("Hotspot Summary")
        st.dataframe(artifacts.hotspot_summary.head(200), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

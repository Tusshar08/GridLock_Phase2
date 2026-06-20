from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOMMENDATION_DIR = PROJECT_ROOT / "outputs" / "recommendations"
MODEL1_METRICS_PATH = PROJECT_ROOT / "outputs" / "model1_v2" / "model1_v2_metrics.json"
MODEL2_METRICS_PATH = PROJECT_ROOT / "outputs" / "model2_v2" / "model2_v2_metrics.json"
MODEL2_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "model2_v2" / "model2_v2_model_comparison.csv"
MODEL2_PER_CLASS_PATH = PROJECT_ROOT / "outputs" / "model2_v2" / "model2_v2_test_per_class_metrics.csv"

RISK_ORDER = ["low", "medium", "high", "critical"]
RISK_COLORS = {
    "low": "#0f766e",
    "medium": "#ca8a04",
    "high": "#ea580c",
    "critical": "#dc2626",
}


st.set_page_config(
    page_title="EventOps AI Command Board",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def require_outputs() -> None:
    required = [
        RECOMMENDATION_DIR / "event_risk_playbooks.csv",
        RECOMMENDATION_DIR / "hotspot_summary.csv",
        RECOMMENDATION_DIR / "recommendation_summary.csv",
        MODEL1_METRICS_PATH,
        MODEL2_METRICS_PATH,
        MODEL2_COMPARISON_PATH,
        MODEL2_PER_CLASS_PATH,
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("Missing required Block 6 inputs.")
        st.code("\n".join(missing))
        st.stop()


def format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def risk_badge(level: str) -> str:
    color = RISK_COLORS.get(level, "#475569")
    return (
        f"<span style='background:{color};color:white;padding:0.25rem 0.55rem;"
        f"border-radius:6px;font-weight:700;text-transform:uppercase;'>{level}</span>"
    )


def metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_playbooks(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    selected_risks = st.sidebar.multiselect(
        "Risk level",
        RISK_ORDER,
        default=RISK_ORDER,
    )

    cause_options = sorted(df["event_cause"].dropna().astype(str).unique())
    selected_causes = st.sidebar.multiselect("Event cause", cause_options)

    corridor_options = sorted(df["corridor"].dropna().astype(str).unique())
    selected_corridors = st.sidebar.multiselect("Corridor", corridor_options)

    only_closure_likely = st.sidebar.checkbox("Closure probability >= 50%")
    only_high_hotspots = st.sidebar.checkbox("High or critical hotspots")
    event_query = st.sidebar.text_input("Search event ID / description")

    filtered = df[df["risk_level"].isin(selected_risks)].copy()
    if selected_causes:
        filtered = filtered[filtered["event_cause"].isin(selected_causes)]
    if selected_corridors:
        filtered = filtered[filtered["corridor"].isin(selected_corridors)]
    if only_closure_likely:
        filtered = filtered[filtered["road_closure_probability"] >= 0.50]
    if only_high_hotspots:
        filtered = filtered[filtered["hotspot_level"].isin(["high", "critical"])]
    if event_query:
        query = event_query.lower().strip()
        haystack = (
            filtered.get("id", "").astype(str)
            + " "
            + filtered.get("description", "").fillna("").astype(str)
        ).str.lower()
        filtered = filtered[haystack.str.contains(query, regex=False)]

    return filtered


def show_command_board(df: pd.DataFrame, filtered: pd.DataFrame) -> None:
    total_events = len(filtered)
    critical_events = int(filtered["risk_level"].eq("critical").sum())
    high_or_critical = int(filtered["risk_level"].isin(["high", "critical"]).sum())
    avg_closure = float(filtered["road_closure_probability"].mean()) if total_events else 0.0
    avg_risk = float(filtered["risk_score"].mean()) if total_events else 0.0

    cols = st.columns(5)
    with cols[0]:
        metric_card("Events", f"{total_events:,}", "after filters")
    with cols[1]:
        metric_card("Critical", f"{critical_events:,}", "immediate command attention")
    with cols[2]:
        metric_card("High+", f"{high_or_critical:,}", "high or critical risk")
    with cols[3]:
        metric_card("Avg Closure", format_pct(avg_closure), "Model 1 probability")
    with cols[4]:
        metric_card("Avg Risk", f"{avg_risk:.1f}", "0-100 playbook score")

    left, right = st.columns([1.1, 1])
    with left:
        risk_counts = (
            filtered["risk_level"]
            .value_counts()
            .reindex(RISK_ORDER, fill_value=0)
            .rename_axis("risk_level")
            .reset_index(name="events")
        )
        st.subheader("Risk Distribution")
        st.bar_chart(risk_counts, x="risk_level", y="events", color="#2563eb")

    with right:
        duration_counts = (
            filtered["predicted_duration_band"]
            .value_counts()
            .reindex(["short", "medium", "long"], fill_value=0)
            .rename_axis("duration_band")
            .reset_index(name="events")
        )
        st.subheader("Predicted Duration")
        st.bar_chart(duration_counts, x="duration_band", y="events", color="#0f766e")

    st.subheader("Priority Queue")
    queue_cols = [
        "id",
        "event_cause",
        "corridor",
        "junction",
        "road_closure_probability",
        "predicted_duration_band",
        "risk_level",
        "risk_score",
        "manpower",
        "agency_alerts",
    ]
    queue_cols = [col for col in queue_cols if col in filtered.columns]
    st.dataframe(
        filtered.sort_values("risk_score", ascending=False)[queue_cols].head(50),
        use_container_width=True,
        hide_index=True,
    )


def show_playbook_explorer(filtered: pd.DataFrame) -> None:
    st.subheader("Playbook Explorer")
    if filtered.empty:
        st.info("No events match the current filters.")
        return

    options = (
        filtered.sort_values("risk_score", ascending=False)
        .assign(
            label=lambda x: (
                x["id"].astype(str)
                + " | "
                + x["risk_level"].astype(str).str.upper()
                + " | "
                + x["event_cause"].astype(str)
                + " | "
                + x["corridor"].astype(str)
            )
        )
    )
    selected_label = st.selectbox("Select event", options["label"].tolist())
    event = options.loc[options["label"].eq(selected_label)].iloc[0]

    top = st.columns([1, 1, 1, 1])
    with top[0]:
        st.markdown(risk_badge(str(event["risk_level"])), unsafe_allow_html=True)
    with top[1]:
        st.metric("Risk Score", f"{float(event['risk_score']):.1f}")
    with top[2]:
        st.metric("Closure Probability", format_pct(float(event["road_closure_probability"])))
    with top[3]:
        st.metric("Duration", str(event["predicted_duration_band"]).title())

    st.divider()
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.write("**Event Context**")
        st.write(f"Cause: `{event.get('event_cause', 'unknown')}`")
        st.write(f"Corridor: `{event.get('corridor', 'unknown')}`")
        st.write(f"Junction: `{event.get('junction', 'unknown')}`")
        st.write(f"Police Station: `{event.get('police_station', 'unknown')}`")
        if "description" in event and pd.notna(event["description"]):
            st.write("Description:")
            st.info(str(event["description"]))

    with col_b:
        st.write("**Recommended Response**")
        st.write(f"Manpower: **{event['manpower']}**")
        st.write(f"Barricading: {event['barricading']}")
        st.write(f"Diversion: {event['diversion']}")
        st.write(f"Equipment: {event['equipment']}")
        st.write(f"Agency Alerts: {event['agency_alerts']}")
        st.write(f"Control Room: {event['control_room']}")

    if {"latitude", "longitude"}.issubset(event.index) and pd.notna(event["latitude"]) and pd.notna(event["longitude"]):
        st.subheader("Location")
        st.map(pd.DataFrame({"lat": [event["latitude"]], "lon": [event["longitude"]]}), zoom=12)


def show_hotspots(hotspots: pd.DataFrame) -> None:
    st.subheader("Hotspot Analytics")
    level_filter = st.multiselect(
        "Hotspot level",
        ["low", "medium", "high", "critical"],
        default=["medium", "high", "critical"],
    )
    table = hotspots[hotspots["hotspot_level"].isin(level_filter)].copy()
    table = table.sort_values("hotspot_score", ascending=False)

    cols = [
        "corridor",
        "junction",
        "police_station",
        "total_events",
        "closure_events",
        "closure_rate",
        "valid_duration_events",
        "long_duration_events",
        "hotspot_score",
        "hotspot_level",
    ]
    st.dataframe(table[cols].head(100), use_container_width=True, hide_index=True)


def get_mappls_static_key() -> str:
    try:
        key = st.secrets.get("MAPPLS_STATIC_KEY", "")
    except Exception:
        key = ""
    return str(key).strip()


@st.cache_data(ttl=300, show_spinner=False)
def check_mappls_sdk_key(mappls_key: str) -> dict:
    sdk_url = f"https://apis.mappls.com/advancedmaps/api/{mappls_key}/map_sdk?layer=vector&v=3.0&callback=initMap"
    try:
        request = urllib.request.Request(
            sdk_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "http://127.0.0.1:8501/",
            },
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            content_type = response.headers.get("content-type", "")
            return {
                "ok": response.status == 200,
                "status": response.status,
                "content_type": content_type,
                "message": "Mappls SDK key accepted.",
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(300).decode("utf-8", errors="replace").replace(mappls_key, "[redacted]")
        return {
            "ok": False,
            "status": exc.code,
            "content_type": exc.headers.get("content-type", ""),
            "message": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "request_failed",
            "content_type": "",
            "message": str(exc).replace(mappls_key, "[redacted]"),
        }


def show_fallback_map(map_df: pd.DataFrame) -> None:
    st.warning("Showing fallback map until the Mappls Web SDK credential is fixed.")
    fallback_df = map_df.rename(columns={"latitude": "lat", "longitude": "lon"})
    st.map(fallback_df[["lat", "lon"]], zoom=11, use_container_width=True)


def show_mappls_heatmap(filtered: pd.DataFrame) -> None:
    st.subheader("Mappls Risk Heatmap")
    mappls_key = get_mappls_static_key()
    if not mappls_key or mappls_key == "paste_your_static_key_here":
        st.warning("Add your Mappls static key to `.streamlit/secrets.toml` to enable this map.")
        st.code('MAPPLS_STATIC_KEY = "your_static_key_here"', language="toml")
        st.info("Use IP / Sub-Net `127.0.0.1` in the Mappls developer console for local testing.")
        return

    required_cols = {"latitude", "longitude", "risk_score", "risk_level", "id", "event_cause"}
    missing = required_cols - set(filtered.columns)
    if missing:
        st.error(f"Missing map columns: {sorted(missing)}")
        return

    map_df = filtered.dropna(subset=["latitude", "longitude"]).copy()
    map_df = map_df[
        map_df["latitude"].between(12.0, 14.0)
        & map_df["longitude"].between(76.0, 78.5)
    ]
    if map_df.empty:
        st.info("No mappable events match the current filters.")
        return

    max_points = st.slider("Max map points", 100, 2000, 600, step=100)
    map_df = map_df.sort_values("risk_score", ascending=False).head(max_points)
    st.caption("If your Mappls credential was created for IP `127.0.0.1`, open Streamlit at `http://127.0.0.1:8501` instead of `localhost:8501`.")

    key_status = check_mappls_sdk_key(mappls_key)
    if not key_status["ok"]:
        st.error("Mappls rejected this key for the Web SDK.")
        st.code(
            f"status: {key_status['status']}\n"
            f"content_type: {key_status['content_type']}\n"
            f"message: {key_status['message']}",
            language="text",
        )
        show_fallback_map(map_df)
        return

    center_lat = float(map_df["latitude"].mean())
    center_lng = float(map_df["longitude"].mean())
    points = []
    for _, row in map_df.iterrows():
        level = str(row.get("risk_level", "low"))
        points.append(
            {
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
                "risk": float(row["risk_score"]),
                "level": level,
                "id": str(row.get("id", "")),
                "cause": str(row.get("event_cause", "unknown")),
                "duration": str(row.get("predicted_duration_band", "unknown")),
                "closure": float(row.get("road_closure_probability", 0.0)),
                "color": RISK_COLORS.get(level, "#475569"),
            }
        )

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta name="viewport" content="initial-scale=1.0, maximum-scale=1.0">
        <style>
          html, body, #map {{ margin:0; padding:0; width:100%; height:620px; background:#e5e7eb; }}
          #map-error {{
            display:none; position:absolute; top:14px; left:14px; right:14px; z-index:1000;
            background:#fff1f2; color:#991b1b; border:1px solid #fecdd3; border-radius:8px;
            padding:10px 12px; font:13px Arial, sans-serif;
          }}
          .legend {{
            position:absolute; bottom:18px; left:18px; background:#fff; padding:10px 12px;
            border-radius:8px; box-shadow:0 1px 6px rgba(15,23,42,.2); font:13px Arial;
            z-index:999;
          }}
          .legend-row {{ display:flex; align-items:center; gap:7px; margin:5px 0; }}
          .dot {{ width:11px; height:11px; border-radius:999px; display:inline-block; }}
        </style>
      </head>
      <body>
        <div id="map"></div>
        <div id="map-error"></div>
        <div class="legend">
          <strong>Risk Level</strong>
          <div class="legend-row"><span class="dot" style="background:{RISK_COLORS['critical']}"></span>Critical</div>
          <div class="legend-row"><span class="dot" style="background:{RISK_COLORS['high']}"></span>High</div>
          <div class="legend-row"><span class="dot" style="background:{RISK_COLORS['medium']}"></span>Medium</div>
          <div class="legend-row"><span class="dot" style="background:{RISK_COLORS['low']}"></span>Low</div>
        </div>
        <script>
          const points = {json.dumps(points)};
          const center = [{center_lat}, {center_lng}];

          function showMapError(message) {{
            const box = document.getElementById('map-error');
            box.style.display = 'block';
            box.innerHTML = message;
          }}

          function addRiskMarkers(map) {{
            points.forEach(function(p) {{
              const radius = Math.max(6, Math.min(20, 5 + p.risk / 6));
              const marker = new mappls.Marker({{
                map: map,
                position: {{ lat: p.lat, lng: p.lng }},
                icon: {{
                  html: '<div style="width:' + (radius * 2) + 'px;height:' + (radius * 2) + 'px;border-radius:999px;background:' + p.color + ';border:2px solid white;box-shadow:0 0 0 2px rgba(15,23,42,.15);opacity:.88"></div>',
                  width: radius * 2,
                  height: radius * 2
                }}
              }});

              const popupHtml =
                '<div style="font:13px Arial; min-width:190px">' +
                '<b>' + p.id + '</b><br>' +
                'Risk: <b>' + p.level.toUpperCase() + '</b> (' + p.risk.toFixed(1) + ')<br>' +
                'Cause: ' + p.cause + '<br>' +
                'Duration: ' + p.duration + '<br>' +
                'Closure probability: ' + (p.closure * 100).toFixed(1) + '%' +
                '</div>';
              marker.addListener('click', function() {{
                new mappls.InfoWindow({{
                  map: map,
                  position: {{ lat: p.lat, lng: p.lng }},
                  content: popupHtml
                }});
              }});
            }});
          }}

          window.initMap = function() {{
            if (!window.mappls || !window.mappls.Map) {{
              showMapError('Mappls SDK loaded, but the map object is unavailable. Check whether this static key is enabled for Web Maps.');
              return;
            }}

            try {{
              const map = new mappls.Map('map', {{
                center: center,
                zoom: 11,
                zoomControl: true,
                geolocation: false
              }});

              map.addListener('load', function() {{
                addRiskMarkers(map);
              }});

              setTimeout(function() {{
                const mapNode = document.getElementById('map');
                if (!mapNode.querySelector('canvas') && !mapNode.querySelector('img')) {{
                  showMapError('Mappls base map did not render. If your key is IP-restricted, open <b>http://127.0.0.1:8501</b>. Also confirm the credential has Web Map / Advanced Maps enabled.');
                }}
              }}, 3500);
            }} catch (err) {{
              showMapError('Mappls map initialization failed: ' + err.message);
            }}
          }}

          const sdk = document.createElement('script');
          sdk.src = 'https://apis.mappls.com/advancedmaps/api/{mappls_key}/map_sdk?layer=vector&v=3.0&callback=initMap';
          sdk.onerror = function() {{
            showMapError('Unable to load Mappls SDK. Check the static key and open the app from <b>http://127.0.0.1:8501</b> if the key is IP-restricted.');
          }};
          document.head.appendChild(sdk);
        </script>
      </body>
    </html>
    """
    components.html(html, height=650, scrolling=False)


def show_model_performance(
    model1_metrics: dict,
    model2_metrics: dict,
    model2_comparison: pd.DataFrame,
    model2_per_class: pd.DataFrame,
) -> None:
    st.subheader("Model 1: Road Closure")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PR-AUC", f"{model1_metrics['test_pr_auc']:.3f}")
    c2.metric("ROC-AUC", f"{model1_metrics['test_roc_auc']:.3f}")
    c3.metric("Recall", f"{model1_metrics['test_recall']:.3f}")
    c4.metric("F1", f"{model1_metrics['test_f1']:.3f}")

    st.subheader("Model 2: Duration Band")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Selected", str(model2_metrics["selected_model"]))
    d2.metric("Macro-F1", f"{model2_metrics['macro_f1']:.3f}")
    d3.metric("Balanced Acc.", f"{model2_metrics['balanced_accuracy']:.3f}")
    d4.metric("Weighted-F1", f"{model2_metrics['weighted_f1']:.3f}")

    left, right = st.columns(2)
    with left:
        st.write("**Model Comparison**")
        st.dataframe(model2_comparison, use_container_width=True, hide_index=True)
    with right:
        st.write("**Duration Per-Class Metrics**")
        st.dataframe(model2_per_class, use_container_width=True, hide_index=True)


def show_pipeline() -> None:
    st.subheader("Pipeline")
    st.markdown(
        """
        ```mermaid
        flowchart LR
          A[Raw Astram Events] --> B[Block 1: Cleaning + Labels]
          B --> C[Block 2: Leakage-Safe Features]
          C --> D[Block 3: Road Closure Model]
          D --> E[Block 4: Duration Band Model]
          E --> F[Block 5: Risk + Playbook Engine]
          F --> G[Block 6: Command Dashboard]
        ```
        """
    )
    st.write("**Run order**")
    st.code(
        "\n".join(
            [
                "notebooks/01_data_cleaning_labels/01_data_cleaning_and_label_creation.ipynb",
                "notebooks/02_feature_engineering/02_feature_engineering_model_ready_datasets.ipynb",
                "notebooks/03_model1_road_closure/03_model1_road_closure_classifier.ipynb",
                "notebooks/04_model2_duration_band/04_model2_duration_band_classifier.ipynb",
                "notebooks/05_recommendation_engine/05_risk_scoring_recommendation_engine.ipynb",
                "app/streamlit_app.py",
            ]
        )
    )


def main() -> None:
    require_outputs()

    playbooks = load_csv(RECOMMENDATION_DIR / "event_risk_playbooks.csv")
    hotspots = load_csv(RECOMMENDATION_DIR / "hotspot_summary.csv")
    summary = load_csv(RECOMMENDATION_DIR / "recommendation_summary.csv")
    model1_metrics = load_json(MODEL1_METRICS_PATH)
    model2_metrics = load_json(MODEL2_METRICS_PATH)
    model2_comparison = load_csv(MODEL2_COMPARISON_PATH)
    model2_per_class = load_csv(MODEL2_PER_CLASS_PATH)

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        .metric-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
            background: #ffffff;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.7rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .metric-caption {
            color: #64748b;
            font-size: 0.78rem;
            min-height: 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("EventOps AI Command Board")
    st.caption("Predict event-driven traffic impact and generate response playbooks.")

    filtered = filter_playbooks(playbooks)

    tabs = st.tabs(["Command Board", "Playbook", "Hotspots", "Mappls Heatmap", "Model Performance", "Pipeline"])
    with tabs[0]:
        show_command_board(playbooks, filtered)
    with tabs[1]:
        show_playbook_explorer(filtered)
    with tabs[2]:
        show_hotspots(hotspots)
    with tabs[3]:
        show_mappls_heatmap(filtered)
    with tabs[4]:
        show_model_performance(model1_metrics, model2_metrics, model2_comparison, model2_per_class)
    with tabs[5]:
        show_pipeline()

    with st.sidebar.expander("Recommendation Summary", expanded=False):
        st.dataframe(summary, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

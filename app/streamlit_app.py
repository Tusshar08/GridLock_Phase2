from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


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

    tabs = st.tabs(["Command Board", "Playbook", "Hotspots", "Model Performance", "Pipeline"])
    with tabs[0]:
        show_command_board(playbooks, filtered)
    with tabs[1]:
        show_playbook_explorer(filtered)
    with tabs[2]:
        show_hotspots(hotspots)
    with tabs[3]:
        show_model_performance(model1_metrics, model2_metrics, model2_comparison, model2_per_class)
    with tabs[4]:
        show_pipeline()

    with st.sidebar.expander("Recommendation Summary", expanded=False):
        st.dataframe(summary, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()

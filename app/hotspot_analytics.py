from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _ensure_text_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str).str.strip().str.lower().replace({"": "unknown"})
    return out


def build_hotspot_summary(history_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = _ensure_text_columns(history_df, ["corridor", "junction", "police_station", "event_cause"])

    if "target_road_closure" not in cleaned.columns:
        cleaned["target_road_closure"] = (cleaned["road_closure_probability"] >= 0.50).astype(int)

    duration_col = "duration_band" if "duration_band" in cleaned.columns else "predicted_duration_band"

    summary = (
        cleaned.groupby(["corridor", "junction", "police_station"], dropna=False)
        .agg(
            total_events=("target_road_closure", "size"),
            closure_events=("target_road_closure", "sum"),
            closure_rate=("target_road_closure", "mean"),
            long_duration_events=(duration_col, lambda s: (s.astype(str).str.lower() == "long").sum()),
            accident_events=("event_cause", lambda s: s.astype(str).str.contains("accident", case=False, na=False).sum()),
            breakdown_events=("event_cause", lambda s: s.astype(str).str.contains("breakdown", case=False, na=False).sum()),
            water_logging_events=("event_cause", lambda s: s.astype(str).str.contains("water", case=False, na=False).sum()),
        )
        .reset_index()
    )

    summary["hotspot_score"] = (
        summary["total_events"].clip(upper=100) * 0.35
        + summary["closure_rate"].fillna(0) * 100 * 0.45
        + summary["long_duration_events"].clip(upper=25) * 0.80
    )
    summary["hotspot_level"] = pd.cut(
        summary["hotspot_score"],
        bins=[-0.01, 15, 35, 60, np.inf],
        labels=["low", "medium", "high", "critical"],
    ).astype(str)

    return summary.sort_values("hotspot_score", ascending=False).reset_index(drop=True)


def build_heatmap_points(
    risk_playbooks: pd.DataFrame,
    max_points: int = 800,
    selected_causes: list[str] | None = None,
    selected_corridors: list[str] | None = None,
    live_event_row: pd.Series | None = None,
) -> list[dict[str, Any]]:
    heat_df = risk_playbooks.copy()
    if selected_causes:
        heat_df = heat_df[heat_df["event_cause"].astype(str).isin(selected_causes)]
    if selected_corridors:
        heat_df = heat_df[heat_df["corridor"].astype(str).isin(selected_corridors)]

    heat_df = heat_df.dropna(subset=["latitude", "longitude"]).copy()
    heat_df = heat_df[
        pd.to_numeric(heat_df["latitude"], errors="coerce").between(12.0, 14.0)
        & pd.to_numeric(heat_df["longitude"], errors="coerce").between(76.0, 78.5)
    ]

    heat_df["heat_weight"] = pd.to_numeric(heat_df.get("road_closure_probability", 0.5), errors="coerce").fillna(0.5).clip(0.05, 1.0)
    if "prob_very_long" in heat_df.columns:
        heat_df["heat_weight"] += pd.to_numeric(heat_df["prob_very_long"], errors="coerce").fillna(0) * 0.7
    if "prob_long" in heat_df.columns:
        heat_df["heat_weight"] += pd.to_numeric(heat_df["prob_long"], errors="coerce").fillna(0) * 0.4
    heat_df["heat_weight"] = heat_df["heat_weight"].clip(0.05, 1.8)

    points = [
        {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "weight": float(row["heat_weight"]),
            "title": str(row.get("id", "Event")),
            "subtitle": f"{row.get('event_cause', 'unknown')} | closure {float(row.get('road_closure_probability', 0)) * 100:.1f}%",
            "is_live": False,
        }
        for _, row in heat_df.sort_values("heat_weight", ascending=False).head(max_points).iterrows()
    ]

    if live_event_row is not None:
        live_weight = float(
            np.clip(
                float(live_event_row.get("road_closure_probability", 0.25))
                + float(live_event_row.get("prob_long", 0)) * 0.4
                + float(live_event_row.get("prob_very_long", 0)) * 0.7,
                0.2,
                2.0,
            )
        )
        points.append(
            {
                "latitude": float(live_event_row.get("latitude", 12.9716)),
                "longitude": float(live_event_row.get("longitude", 77.5946)),
                "weight": live_weight,
                "title": str(live_event_row.get("id", "Live event")),
                "subtitle": f"live | closure {float(live_event_row.get('road_closure_probability', 0)) * 100:.1f}%",
                "is_live": True,
            }
        )

    return points

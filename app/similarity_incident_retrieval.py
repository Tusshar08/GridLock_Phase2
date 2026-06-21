from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _build_text_embedding(series: pd.Series) -> np.ndarray:
    text = series.fillna("").astype(str)
    return np.vstack(
        [
            text.str.len().to_numpy(),
            text.str.split().str.len().fillna(0).to_numpy(),
            text.str.count(r"[A-Za-z]").to_numpy(),
            text.str.count(r"[^\x00-\x7F]").to_numpy(),
        ]
    ).T.astype(float)


def build_similarity_store(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, NearestNeighbors, StandardScaler, PCA | None, list[str]]:
    candidate_struct_cols = [
        "latitude",
        "longitude",
        "start_hour",
        "start_dayofweek",
        "start_month_number",
        "report_lag_minutes_clipped",
        "distance_to_city_center_km",
        "road_closure_probability",
    ]
    struct_cols = [c for c in candidate_struct_cols if c in df.columns]

    x_struct = df[struct_cols].copy() if struct_cols else pd.DataFrame(index=df.index)
    for col in struct_cols:
        x_struct[col] = pd.to_numeric(x_struct[col], errors="coerce")
    x_struct = x_struct.fillna(x_struct.median(numeric_only=True)).fillna(0.0)

    scaler = StandardScaler()
    struct_scaled = scaler.fit_transform(x_struct) if struct_cols else np.zeros((len(df), 0))

    text_emb = _build_text_embedding(df.get("description", pd.Series("", index=df.index)))
    text_pca: PCA | None = None

    if len(df) < 2 or float(np.nan_to_num(np.var(text_emb, axis=0), nan=0.0).sum()) <= 0.0:
        text_reduced = np.zeros((len(df), 1), dtype=float)
    else:
        n_text_components = min(4, text_emb.shape[1], max(1, len(df) - 1))
        text_pca = PCA(n_components=n_text_components, random_state=42)
        text_reduced = text_pca.fit_transform(text_emb)

    vectors = np.hstack([struct_scaled, text_reduced]).astype("float32")
    norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    vectors_norm = vectors / norm

    index = NearestNeighbors(metric="cosine", algorithm="auto")
    index.fit(vectors_norm)

    meta_cols = [
        c
        for c in [
            "id",
            "event_type",
            "event_cause",
            "corridor",
            "junction",
            "police_station",
            "zone",
            "veh_type",
            "start_datetime",
            "latitude",
            "longitude",
            "duration_band",
            "road_closure_probability",
            "predicted_duration_band",
            "prediction_confidence",
            "risk_level",
            "risk_score",
            "manpower",
            "barricading",
            "diversion",
            "equipment",
            "agency_alerts",
        ]
        if c in df.columns
    ]
    meta = df[meta_cols].copy()
    return meta, vectors_norm, index, scaler, text_pca, struct_cols


def query_similar_incidents(
    query_event: pd.Series,
    similarity_meta: pd.DataFrame,
    similarity_index: NearestNeighbors,
    scaler: StandardScaler,
    text_pca: PCA | None,
    struct_cols: list[str],
    top_k: int = 5,
) -> pd.DataFrame:
    query_struct = pd.DataFrame([query_event]).reindex(columns=struct_cols, fill_value=0.0)
    for col in query_struct.columns:
        query_struct[col] = pd.to_numeric(query_struct[col], errors="coerce")
    query_struct = query_struct.fillna(0.0)

    q_struct_scaled = scaler.transform(query_struct) if struct_cols else np.zeros((1, 0))
    q_text = _build_text_embedding(pd.Series([query_event.get("description", "")]))
    if text_pca is None:
        q_text_reduced = np.zeros((1, 1), dtype=float)
    else:
        q_text_reduced = text_pca.transform(q_text)

    q_vec = np.hstack([q_struct_scaled, q_text_reduced]).astype("float32")
    q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)

    n_neighbors = min(len(similarity_meta), top_k)
    if n_neighbors == 0:
        return pd.DataFrame()

    dist, idx = similarity_index.kneighbors(q_vec, n_neighbors=n_neighbors)
    selected = similarity_meta.iloc[idx[0]].copy().reset_index(drop=True)
    selected["similarity_score"] = [round(1 - d, 4) for d in dist[0]]
    return selected

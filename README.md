# RoadGuard AI

RoadGuard AI is a Streamlit dashboard for traffic event response planning. It uses trained road-closure and clearance-time models, hotspot analytics, and similar-incident retrieval to turn a new traffic report into a suggested field response plan.

## What It Does

- Estimates overall event risk.
- Predicts road closure likelihood.
- Estimates expected clearance-time band.
- Suggests manpower, barricading, diversion, control-room updates, and equipment.
- Shows similar past incidents for context.
- Displays a city hotspot view using Mappls when credentials are available.

## Repository Layout

```text
app/
  streamlit_app.py                         # RoadGuard AI dashboard
  pipeline_engine.py                       # Loads artifacts and runs prediction workflow
  hotspot_analytics.py                     # Hotspot summary and heatmap point generation
  risk_scoring_recommendation_engine.py    # Risk scoring and response recommendations
  similarity_incident_retrieval.py         # Similar historical incident lookup

data/
  Astram event data_anonymized ... .csv    # Source event data

notebooks/
  EDA/                                     # Exploratory analysis
  feature_engineering/                     # Feature creation notebooks
  model1/                                  # Road closure model notebooks
  model2/                                  # Clearance-time model notebooks

outputs/
  features/                                # Model-ready feature tables
  model_road_closure/                      # Model 1 artifacts and handoff data
  model_duration_band/                     # Model 2 artifacts and handoff data
```

Generated app runtime outputs under `outputs/recommendations/` and `outputs/similarity/` are ignored by git.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Mappls Credentials

Copy the example secrets file:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Then fill in one of the following:

- `CLIENT_ID` and `CLIENT_SECRET` for OAuth token generation.
- Or `MAPPLS_STATIC_KEY` / `MAPPLS_ACCESS_TOKEN` if you already have a valid Web SDK token.

The app can still run without Mappls credentials, but the hotspot map will fall back to Streamlit's basic map view.

## Run the App

From the project root:

```bash
streamlit run app/streamlit_app.py
```

If you are using the local virtual environment directly:

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Deploy on Render

This repo includes a Render Blueprint file at `render.yaml`.

1. Push the repository to GitHub.
2. In Render, choose `New` -> `Blueprint`.
3. Connect the GitHub repository.
4. Render will read `render.yaml` and create the `roadguard-ai` web service.
5. Add the required environment variables in Render:

```text
CLIENT_ID
CLIENT_SECRET
```

Optional if you have a valid static Mappls Web SDK token:

```text
MAPPLS_STATIC_KEY
```

The Render service uses:

```bash
pip install -r render-requirements.txt
```

And starts the app with:

```bash
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true
```

If you deploy manually instead of using the Blueprint, create a Python Web Service and use the same build and start commands above.

## Main Workflow

1. Open the `Response Plan` tab.
2. Enter the traffic event details.
3. Click `Get Response Plan`.
4. Review the risk metrics, recommended field plan, priority actions, and similar past events.
5. Use the `Hotspot View` tab to inspect high-attention areas across the city.

## Notes

- Keep `.streamlit/secrets.toml` private. It is ignored by git.
- Large generated outputs and local runtime files should not be committed.
- The app expects the trained model artifacts under `outputs/model_road_closure/` and `outputs/model_duration_band/`.

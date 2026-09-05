# 🌫️ 10Pearls AQI Predictor

A serverless, end-to-end machine learning system that forecasts the **US EPA Air Quality Index** for **Lahore, Pakistan** at **24, 48, and 72 hours ahead**.

**Live dashboard:** [pearl-aqi-predictor-mahmedanjum.streamlit.app](https://pearl-aqi-predictor-mahmedanjum.streamlit.app/)

**Full technical report:** [`10Pearls_AQI_Predictor_Report.pdf`](./10Pearls_AQI_Predictor_Report.pdf) (18 pages — data, methodology, full results, and 8 real production incidents found and fixed)

Built by **Muhammad Ahmed Anjum** — 10Pearls Data Science Internship Programme

---

## Table of contents

* [What it does](#what-it-does)
* [Results](#results)
* [Architecture](#architecture)
* [Tech stack](#tech-stack)
* [Repository structure](#repository-structure)
* [Running locally](#running-locally)
* [How the forecast actually works](#how-the-forecast-actually-works)
* [Key design decisions](#key-design-decisions)
* [What broke, and how it was found](#what-broke-and-how-it-was-found)
* [Limitations](#limitations)
* [What I'd do next](#what-id-do-next)

---

## What it does

* **Fetches** hourly weather and pollutant data from the OpenWeather Air Pollution API
* **Computes** AQI from raw pollutant concentrations using the US EPA breakpoint formula
* **Engineers features**: time-based (hour, day, cyclically encoded), AQI lags (1–24h), rolling mean/std (3/24/72h)
* **Stores** everything in a Hopsworks feature store (~49,500 hourly rows, Nov 2020 → present)
* **Trains and compares** three model families — Ridge Regression, Random Forest, and a neural network — against a naive "assume no change" baseline
* **Registers** the winning model to the Hopsworks Model Registry
* **Serves** live 24h/48h/72h forecasts through a Streamlit dashboard and a REST API, with SHAP explainability and hazardous-AQI alerts
* **Automates** the entire loop — hourly data collection, daily retraining, weekly keepalive — via GitHub Actions, requiring zero manual intervention

---

## Results

Three model families were evaluated identically (same features, same chronological split, same evaluation) against a persistence baseline:

| Horizon | Persistence R² | Random Forest R² | Neural Net R² | **Ridge R² (deployed)** |
| ------- | -------------- | ---------------- | ------------- | ----------------------- |
| 24h     | 0.804          | 0.813            | 0.810         | **0.824**               |
| 48h     | 0.695          | 0.721            | 0.719         | **0.740**               |
| 72h     | 0.632          | 0.693            | 0.674         | **0.694**               |

**Ridge Regression wins at every horizon** and is the deployed model. The neural network — a small Keras MLP — was consistently the weakest of the three, a genuine and informative negative result rather than a shortfall in the experiment (small MLPs commonly underperform simpler models on modest tabular datasets like this one).

All results, the full methodology, and the honest story of what *didn't* work on the first attempt (a naive model looked good — R² = 0.93 — by silently answering an easier question than the one asked) are in the [technical report](./10Pearls_AQI_Predictor_Report.pdf).

---

## Architecture

```text
                 ┌───────────────────┐
  OpenWeather ──▶│  Feature Pipeline  │──▶  Hopsworks Feature Store
    (hourly)      │  (GitHub Actions)  │      (lahore_aqi_features)
                 └───────────────────┘
                           │
                           ▼
                 ┌───────────────────┐      ┌────────────────┐
                 │ Training Pipeline  │◀─────│  ~49,500 rows   │
                 │  (GitHub Actions,  │      │  Nov 2020 → now │
                 │       daily)       │      └────────────────┘
                 └───────────────────┘
                           │
                           ▼
                  Hopsworks Model Registry
                   (lahore_aqi_forecast_model)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Streamlit Dashboard        FastAPI Service
       (charts, SHAP, alerts)   (/health /current /forecast)
```

The feature pipeline pulls a **rolling 48-hour window** on every run, not just the latest hour — this makes it self-healing against GitHub Actions' documented scheduling delays. If a run is skipped or delayed, the next successful run automatically backfills the gap (Hopsworks upserts by timestamp, so re-sending overlapping hours is safe).

---

## Tech stack

**Python** · **scikit-learn** · **TensorFlow/Keras** · **Hopsworks** (feature store + model registry) · **GitHub Actions** (CI/CD) · **Streamlit** + **Plotly** (dashboard) · **FastAPI** (REST API) · **SHAP** (explainability) · **OpenWeather API** (data source) · **Git**

---

## Repository structure

```text
feature_pipeline/
├── fetch_features.py         # Hourly job — self-healing 48h rolling window
├── backfill.py                # One-time historical load (~5.7 years, ~30-day chunks)
├── aqi_utils.py                # EPA breakpoint formula (PM2.5 → AQI)
├── upload_to_hopsworks.py     # Bulk backfill upload
├── dedupe.py                   # De-duplicates the local CSV before upload
└── check_data_quality.py      # Value-range, coverage, and duplicate checks

training_pipeline/
├── train_model.py             # Feature engineering, delta target, model training/comparison
├── register_model.py           # Registers the winning model + metrics to Hopsworks
└── compare_dl_model.py         # Standalone 3-way comparison, incl. the neural network

app/
├── streamlit_app.py            # Live dashboard
└── aqi_style.py                 # Theming and EPA category colours

api/
├── main.py                      # FastAPI service (/health, /current, /forecast)
└── requirements.txt

.github/workflows/
├── hourly_feature_pipeline.yml # Runs every hour (offset to :17 to avoid GitHub's :00 congestion)
├── daily_training_pipeline.yml # Runs once a day
└── keepalive.yml                # Weekly commit — prevents GitHub Actions dormancy

10Pearls_AQI_Predictor_Report.pdf   # Full technical report
requirements.txt                     # Root deps (used by the Streamlit Cloud deployment)
```

---

## Running locally

```bash
git clone https://github.com/ahmedquresshi579/Pearl-AQI-Predictor.git
cd Pearl-AQI-Predictor

python -m venv venv

venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file with the following (needed in `feature_pipeline/`, `training_pipeline/`, `app/`, and `api/` — each service reads its own):

```env
OPENWEATHER_API_KEY=your_openweather_key
HOPSWORKS_API_KEY=your_hopsworks_key
```

### First-time setup

Populate history and train the model:

```bash
python feature_pipeline/backfill.py
python feature_pipeline/dedupe.py
python feature_pipeline/upload_to_hopsworks.py

python training_pipeline/train_model.py
python training_pipeline/register_model.py
```

### Run the apps

```bash
streamlit run app/streamlit_app.py
# dashboard → http://localhost:8501

uvicorn api.main:app --reload
# API → http://localhost:8000/docs
```

From here, `.github/workflows/` takes over automatically once pushed to GitHub: hourly data collection, daily retraining, and a weekly keepalive commit — no further manual steps needed.

---

## How the forecast actually works

The model does **not** predict the AQI level directly. It predicts the **change** in AQI over the next 24/48/72 hours, and the final forecast is anchored to the current reading:

```text
forecast = current_aqi + blend_weight × model.predict(features)
```

with `blend_weight = 0.5`.

This single design choice is the difference between a forecast that works and one that doesn't — see [Results](#results) and the report's §4 and §6.1 for the full story of why a level-prediction model collapsed to near-zero skill at 72 hours, and why this fixed it.

**Feature set (24 features):** current pollutant readings, cyclically-encoded hour, day of week, AQI lags at 1/2/3/6/12/24 hours, and rolling mean/std of AQI over 3/24/72-hour windows (each window shifted by one period so a row never includes its own value).

---

## Key design decisions

* **Chronological train/test split, never random** — a random split on time-series data lets a model evaluate on rows whose near-identical neighbours it trained on, inflating apparent accuracy.
* **Timestamp-matched joins, never positional row shifts** — with ~2% of hours missing from the archive, a positional `.shift(N)` would silently misalign a row with the wrong past/future value across a gap. This governs every lag, rolling-window, and label-matching operation in the codebase.
* **Self-healing hourly fetch** — pulls a rolling 48-hour window every run rather than a single reading, so a delayed or skipped GitHub Actions run (a documented platform limitation, especially at the top of the hour) corrects itself automatically on the next successful run.
* **Cron offset from `:00` to `:17`** — GitHub's own guidance: jobs scheduled at the exact top of the hour compete with the platform's highest load.
* **Every model is scored against a persistence baseline** ("assume no change"), at every horizon, on the same rows — an R² number in isolation can be misleading; beating persistence is the bar that actually matters.
* **Models load lazily in the API**, not at startup — a transient Hopsworks outage doesn't stop the service from booting; `/health` never contacts Hopsworks directly, so a health probe can't itself become the load that takes the service down.
* **Automatic retries with a graceful fallback in the dashboard** — Hopsworks' free-tier Feature Query Service has occasional transient outages; the dashboard retries a few times with a short delay before showing a clean "temporarily unavailable" message instead of a raw stack trace.
* **A weekly keepalive workflow** exists because this project is evaluated by submitted link with no fixed date — GitHub disables scheduled workflows after 60 days without a commit, so the keepalive job makes a trivial weekly commit specifically to prevent that.

---

## What broke, and how it was found

Eight real production issues were found and fixed during development, several of a kind that reported success while silently doing nothing useful. Full details, with exact root causes and fixes, are in the report's §8 — summarized here:

1. **The hourly fetch job ran green for weeks while writing zero rows** — it computed features correctly but never actually called `fg.insert()`.
2. **Intermittent schema rejection** — OpenWeather occasionally returns an exact whole number (e.g. `no: 0`), which pandas infers as an integer for a single-row frame; the feature group's `double` schema then rejected it.
3. **Permanent data gaps from single-hour fetches** — any scheduling delay meant that hour's data was lost forever, with no recovery. Fixed by the rolling 48h window.
4. **The dashboard silently served a stale model for weeks** — `get_model(version=None)` doesn't mean "latest" in the Hopsworks API; it defaults to version 1.
5. **Wrong metrics attached to the deployed model** — a metadata edit only updated the model's registry *name*, not its attached metrics dict.
6. **Raw HTML rendered as literal text in the dashboard** — indented multi-line f-strings inside `st.markdown()` were being parsed as Markdown code blocks.
7. **The entire dashboard existed only on a local machine for weeks** — never actually committed to the repository.
8. **A 207 MB model file** — reduced to ~10 MB by tuning Random Forest hyperparameters, with a negligible accuracy cost.

Every one of these was caught by directly checking the thing that should have changed (a row count, a rendered value, a file's presence in the repo) — never by trusting a green checkmark.

---

## Limitations

* **The AQI is an hourly proxy, not the official EPA index** — EPA breakpoints are applied to a single hourly reading rather than their proper multi-hour averaging window, since that's the only cadence available from a free data source. Disclosed and consistent as a forecasting target; not a figure to publish as official EPA AQI for a given hour.
* **Single, modeled data source** — OpenWeather's pollutant data is model-derived, not a direct ground-sensor reading, and was observed to differ from multi-station city aggregates (aqicn.org, IQAir) by 30–40 AQI points at the same hour during validation. Normal cross-source variance, not a defect — but the forecast should be read as an estimate for one coordinate, not a citywide average.
* **History capped at November 2020** — OpenWeather's free-tier pollution archive doesn't reach further back.
* **GitHub Actions scheduling is not guaranteed exact** — mitigated (cron offset, self-healing fetch) but not eliminated.
* **Blend weight is fixed at 0.5**, not tuned per horizon on a validation set.
* **Single city** — nothing here has been tested outside Lahore.
* **Evaluation timing is unknown** (link-only submission, no fixed date) — the keepalive workflow and self-healing fetch specifically address the risk of the system looking stale or broken by the time it's actually reviewed.

Full discussion in the report, §9.

---

## What I'd do next

* Add an ARIMA baseline trained only on AQI's own history, completing the brief's statistical-to-deep-learning model range (the neural network side is already done — see [Results](#results)).
* Roll pollutant concentrations to their proper EPA averaging windows and re-measure against both the current and corrected target.
* Tune the blend weight per horizon on a held-out validation slice rather than using a fixed 0.5 for all three.
* Add walk-forward evaluation (retrain at successive time origins, score only the following window) to better match how the daily-retraining deployment actually behaves in production.

---

*10Pearls AQI Predictor · 10Pearls Data Science Internship Programme*

# Iris Classifier: MLOps Project Demo Notes

## 1. Why this is an MLOps project (not just an ML model)

An ML project ends when a model reaches good accuracy. An **MLOps** project covers the whole
lifecycle: the model is trained **reproducibly**, **tracked**, **versioned**, **tested**, **packaged**,
**deployed as a service** and **delivered automatically**. This project covers each of those stages
with standard industry tools. The model is kept simple (Iris + Logistic Regression) on purpose, so
the demo can focus on the pipeline around it.

## 2. MLOps lifecycle → what this project implements

| MLOps practice | What it means | Where it is in this project |
|---|---|---|
| Experiment tracking | Every training run's parameters, metrics and artifacts are recorded | `src/train.py` logs `C`, `max_iter`, `accuracy`, `f1_macro` and the model to **MLflow** |
| Model registry & versioning | Trained models are stored as numbered versions | Each run registers a new version of `iris-classifier` in the **MLflow Model Registry** |
| Reproducibility | The same code and data give the same result | Fixed `random_state`, stratified split, pinned `requirements.txt`, Docker image |
| Model serving | The model is exposed as an API that other systems can call | **FastAPI** `POST /predict` and `GET /health` in `src/serve.py` |
| Input validation / API contract | Bad requests are rejected, and the schema is documented | **Pydantic** schema (values must be > 0); invalid input → HTTP 422; auto-generated Swagger at `/docs` |
| Decoupled frontend | The UI is separate from the model and only talks to the API | **Gradio** `src/ui.py` calls `/predict` over HTTP |
| Resilience / graceful degradation | The system still works when a component fails | The UI falls back to local `model.pkl` when the API is down and shows which mode it used |
| Automated testing | Model quality and API behaviour are checked in code | `pytest`: accuracy ≥ 0.9 quality gate, API tests, UI build test, fallback test |
| Containerization | Same runtime everywhere (laptop, CI, cloud) | `Dockerfile` trains the model at build time and serves it with Uvicorn |
| CI/CD | Every change is tested and packaged automatically | **GitHub Actions**: install → train → test → docker build → smoke-test `/predict` in the container |
| Configuration via environment | The same code runs in different environments | `MLFLOW_TRACKING_URI`, `MODEL_PATH`, `API_URL` env vars |

## 3. Architecture

```
            ┌──────────────┐   HTTP POST /predict   ┌──────────────┐   loads   ┌─────────────────────┐
  User ───▶ │  Gradio UI   │ ─────────────────────▶ │ FastAPI API  │ ────────▶ │ model.pkl           │
            │  :7860       │ ◀───────────────────── │ :8000        │           │ (from training run) │
            └──────────────┘   {"species": ...}     └──────────────┘           └─────────▲───────────┘
                                                                                         │
            ┌─────────────────────────────────────────────────────┐                      │
            │ src/train.py ──▶ MLflow tracking + Model Registry   │ ─────────────────────┘
            │                  (sqlite:///mlflow.db, UI on :5000) │
            └─────────────────────────────────────────────────────┘
  GitHub Actions: train → pytest → docker build → container smoke test (on every push/PR)
```

This is the production pattern **frontend → inference API → model registry**: each layer can be
changed, scaled or redeployed on its own.

## 4. Demo script (about 5 minutes)

1. **Train**: `python src/train.py` → shows accuracy and "Created version N of model 'iris-classifier'".
   Run it again with `--C 0.1` to create a second, comparable run.
2. **MLflow UI** (http://localhost:5000): compare the two runs' params/metrics side by side, then open
   *Models → iris-classifier* to show the version history. *Talking point: tracking + versioning.*
3. **FastAPI Swagger** (http://localhost:8000/docs): run `POST /predict`, then send invalid input to
   show the 422 error. *Talking point: the model as a service with a validated contract.*
4. **Gradio UI** (http://localhost:7860): click an example and press Predict; the mode box says
   *API mode*. Stop Uvicorn and predict again; it now says *Local fallback mode*.
   *Talking point: decoupled frontend + resilience.*
5. **GitHub Actions** (the PR's Checks tab): show the green pipeline: train → test → Docker build →
   container smoke test. *Talking point: CI/CD and automated quality gates.*
6. **Docker**: `docker build -t iris-mlops . && docker run -p 8000:8000 iris-mlops`.
   *Talking point: a portable, reproducible deployment artifact.*

## 5. Likely examiner questions

- **"Why such a simple model?"** The project is about the lifecycle, not the algorithm. Swapping in
  a different model only changes `train.py`; tracking, serving, tests, CI and Docker stay the same.
- **"What makes it reproducible?"** Pinned dependencies, a fixed random seed, a Docker image that
  trains and serves in one defined environment, and CI that rebuilds everything from scratch.
- **"How do you know a bad model won't be deployed?"** The test suite has an accuracy ≥ 0.9 gate,
  and CI must pass before the Docker image is considered good.
- **"Why a separate UI and API?"** In production the frontend, the inference service and the model
  lifecycle are owned and scaled separately. The UI only depends on the API contract.

## 6. Next steps toward production (shows awareness of the full lifecycle)

- Serve directly from the registry (`models:/iris-classifier@production`) instead of `model.pkl`.
- Push the Docker image to a registry (GHCR/ACR) and deploy it automatically (CD) to a cloud service.
- Monitoring: log requests and predictions, track latency, detect data drift (e.g. Evidently).
- Add data versioning (DVC), and scheduled retraining when drift is detected.

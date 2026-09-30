# mlops-project — Iris classifier MLOps mini project

An end-to-end MLOps demo on the Iris dataset:

| Layer | Tool | File |
|---|---|---|
| Training + experiment tracking + model registry | scikit-learn, MLflow | `src/train.py` |
| Inference API | FastAPI + Uvicorn | `src/serve.py` |
| User-facing web UI | Gradio | `src/ui.py` |
| Tests | pytest | `tests/test_app.py` |
| CI | GitHub Actions | `.github/workflows/ci.yml` |
| Packaging | Docker | `Dockerfile` |

The Gradio UI does **not** load the model itself: it calls the FastAPI `/predict` endpoint over HTTP.
This mirrors a real production pattern — **frontend → inference API → model registry** — where the
UI, the serving layer, and the model lifecycle are decoupled and can be deployed/scaled independently.
If the API is unreachable, the UI falls back to loading `model.pkl` locally and shows which mode was used.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py            # optional: --C 0.5 --max-iter 300
```

This logs params/metrics/model to MLflow (`sqlite:///mlflow.db`), registers a new version of the
`iris-classifier` model in the MLflow Model Registry, and writes `model.pkl` for serving.

## Exam demo flow

Run each in its own terminal from the repo root:

1. **MLflow UI** — experiments, runs, metrics, registered model versions

   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
   ```

   Open http://localhost:5000

2. **FastAPI inference API** — Swagger UI at `/docs` is a free interactive UI

   ```bash
   uvicorn src.serve:app --port 8000
   ```

   Open http://localhost:8000/docs and try `POST /predict`, or:

   ```bash
   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
   # {"species":"setosa","class_id":0}
   ```

3. **Gradio web UI** — sliders for the four features, one-click examples

   ```bash
   python src/ui.py
   ```

   Open http://localhost:7860, move the sliders (or click an example) and press **Predict**.
   The "Prediction mode" box shows `API mode` when served by FastAPI; stop the API to see the
   `Local fallback mode`.

   ![Gradio UI screenshot](docs/gradio-ui.png)
   <!-- TODO: add screenshot of the Gradio UI at docs/gradio-ui.png -->

Environment variables: `API_URL` (UI → API, default `http://localhost:8000`), `MODEL_PATH`
(default `model.pkl`), `MLFLOW_TRACKING_URI` (default `sqlite:///mlflow.db`).

## Tests

```bash
pytest -v
```

Covers training accuracy, `/health`, `/predict` (valid + invalid input), that the Gradio interface
builds without launching a server, and the UI's local-fallback path.

## Docker

The image trains the model at build time and serves the API on port 8000:

```bash
docker build -t iris-mlops .
docker run -p 8000:8000 iris-mlops
```

To also run the Gradio UI in the container, switch to the commented-out alternative `CMD` in the
`Dockerfile` (runs both the API and the UI; port 7860 is exposed):

```bash
docker run -p 8000:8000 -p 7860:7860 iris-mlops
```

Alternatively keep the API in Docker and run the UI on the host with `python src/ui.py` — it talks
to the container via `http://localhost:8000`.

## CI

GitHub Actions runs on every push to `main` and on pull requests: install dependencies → train →
pytest → build the Docker image → smoke-test `/predict` in the running container.

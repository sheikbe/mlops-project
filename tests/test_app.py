import importlib

import gradio as gr
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory, monkeypatch_session):
    tmp = tmp_path_factory.mktemp("artifacts")
    model_path = tmp / "model.pkl"
    monkeypatch_session.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp / 'mlflow.db'}")
    monkeypatch_session.setenv("MODEL_PATH", str(model_path))
    from src import train

    importlib.reload(train)
    accuracy = train.train()
    return model_path, accuracy


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def client(trained_model):
    from src import serve

    importlib.reload(serve)
    return TestClient(serve.app)


def test_training_accuracy(trained_model):
    _, accuracy = trained_model
    assert accuracy >= 0.9


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_setosa(client):
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"species": "setosa", "class_id": 0}


def test_predict_rejects_invalid_input(client):
    resp = client.post("/predict", json={"sepal_length": "abc"})
    assert resp.status_code == 422


def test_gradio_interface_builds():
    from src import ui

    assert isinstance(ui.demo, gr.Interface)


def test_ui_falls_back_to_local_model(trained_model, monkeypatch):
    from src import ui

    monkeypatch.setattr(ui, "API_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(ui, "MODEL_PATH", str(trained_model[0]))
    species, mode = ui.predict(5.1, 3.5, 1.4, 0.2)
    assert species == "setosa"
    assert mode.startswith("Local fallback mode")

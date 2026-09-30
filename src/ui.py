import os

import gradio as gr
import joblib
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
SPECIES = ["setosa", "versicolor", "virginica"]


def predict(sepal_length, sepal_width, petal_length, petal_width):
    features = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        resp = requests.post(f"{API_URL}/predict", json=features, timeout=3)
        resp.raise_for_status()
        return resp.json()["species"], f"API mode: served by {API_URL}/predict"
    except requests.RequestException:
        if not os.path.exists(MODEL_PATH):
            raise gr.Error(f"API unreachable and no local model at {MODEL_PATH}")
        class_id = int(joblib.load(MODEL_PATH).predict([list(features.values())])[0])
        return SPECIES[class_id], f"Local fallback mode: API unreachable, used {MODEL_PATH}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.8, step=0.1, label="Sepal length (cm)"),
        gr.Slider(2.0, 4.5, value=3.0, step=0.1, label="Sepal width (cm)"),
        gr.Slider(1.0, 7.0, value=4.3, step=0.1, label="Petal length (cm)"),
        gr.Slider(0.1, 2.5, value=1.3, step=0.1, label="Petal width (cm)"),
    ],
    outputs=[gr.Textbox(label="Predicted species"), gr.Textbox(label="Prediction mode")],
    examples=[[5.1, 3.5, 1.4, 0.2], [5.9, 2.8, 4.3, 1.3], [6.9, 3.1, 5.4, 2.1]],
    title="Iris Species Classifier",
    description="Gradio frontend → FastAPI inference API → MLflow-tracked model",
    submit_btn="Predict",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()

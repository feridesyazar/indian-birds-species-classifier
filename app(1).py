import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image


# =========================================================
# CLASS LABELS
# =========================================================

BIRD_CLASSES = [
    "Asian-Green-Bee-Eater",
    "Brown-Headed-Barbet",
    "Cattle-Egret",
    "Common-Kingfisher",
    "Common-Myna",
    "Common-Rosefinch",
    "Common-Tailorbird",
    "Coppersmith-Barbet",
    "Forest-Wagtail",
    "Gray-Wagtail",
    "Hoopoe",
    "House-Crow",
    "Indian-Grey-Hornbill",
    "Indian-Peacock",
    "Indian-Pitta",
    "Indian-Roller",
    "Jungle-Babbler",
    "Northern-Lapwing",
    "Red-Wattled-Lapwing",
    "Ruddy-Shelduck",
    "Rufous-Treepie",
    "Sarus-Crane",
    "White-Breasted-Kingfisher",
    "White-Breasted-Waterhen",
    "White-Wagtail"
]


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATHS = [
    "indian_birds_resnet50v2.keras",
    "src/indian_birds_resnet50v2.keras",
    "model.keras",
    "model.h5"
]


def find_model_path():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Model file not found. Please upload "
        "indian_birds_resnet50v2.keras to the same folder as app.py."
    )


# =========================================================
# LOAD MODEL
# =========================================================

model = None


def load_model():
    global model

    if model is None:
        model_path = find_model_path()
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

    return model


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image: Image.Image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.resnet_v2.preprocess_input(
        img_array
    )

    return img_array


# =========================================================
# PREDICTION
# =========================================================

def predict_bird(image):

    if image is None:
        return {}, "Please upload a bird image first."

    try:

        loaded_model = load_model()

        processed_image = preprocess_image(image)

        predictions = loaded_model.predict(
            processed_image,
            verbose=0
        )[0]

        if len(predictions) != len(BIRD_CLASSES):
            return {}, (
                f"Class mismatch: model returned "
                f"{len(predictions)} classes, "
                f"but {len(BIRD_CLASSES)} classes are defined."
            )

        top_indices = predictions.argsort()[-5:][::-1]

        label_results = {
            BIRD_CLASSES[idx]: float(predictions[idx])
            for idx in top_indices
        }

        best_idx = top_indices[0]
        best_label = BIRD_CLASSES[best_idx]
        best_score = float(predictions[best_idx])

        note = ""

        if best_label in [
            "Jungle-Babbler",
            "Rufous-Treepie"
        ]:
            note = (
                "These species can appear visually similar. "
                "Tail structure and feather texture are useful differentiators."
            )

        elif "Kingfisher" in best_label:
            note = (
                "Beak shape and chest coloration are distinctive "
                "features for Kingfisher species."
            )

        elif best_label == "Indian-Peacock":
            note = (
                "Distinctive plumage and crest morphology were detected."
            )

        result_text = (
            f"Predicted Species: {best_label}\n"
            f"Confidence Score: {best_score * 100:.2f}%"
        )

        if note:
            result_text += f"\n\n{note}"

        return label_results, result_text

    except Exception as e:
        return {}, f"Prediction error: {str(e)}"


def clear_app():
    return None, {}, ""


# =========================================================
# DESIGN
# =========================================================

CUSTOM_CSS = """
:root {
    --forest-dark: #244b35;
    --forest: #2f6b3f;
    --leaf: #5b9a68;
    --leaf-soft: #dcebdc;
    --cream: #fffaf2;
    --saffron: #f4a340;
    --text: #1f2d22;
    --muted: #617064;
    --border: #d7e6d8;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    min-height: 100vh;

    background:
        radial-gradient(
            circle at 8% 8%,
            rgba(244,163,64,0.15),
            transparent 24%
        ),
        radial-gradient(
            circle at 92% 12%,
            rgba(80,145,93,0.18),
            transparent 28%
        ),
        linear-gradient(
            145deg,
            #f7fbf5 0%,
            #eef7ed 48%,
            #fdf8ef 100%
        );
}

#bird-shell {
    padding: 30px 20px 38px 20px;
}

/* HERO */
#bird-header {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            #244b35 0%,
            #357348 55%,
            #4f8d5b 100%
        );

    border-radius: 26px;
    padding: 32px 34px;
    margin-bottom: 24px;

    box-shadow:
        0 18px 46px
        rgba(36,75,53,0.20);
}

#bird-header::before {
    content: "";
    position: absolute;
    width: 250px;
    height: 250px;
    right: -80px;
    top: -100px;

    border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.10);
}

#bird-header::after {
    content: "";
    position: absolute;
    width: 140px;
    height: 140px;
    right: 70px;
    bottom: -85px;

    border-radius: 50%;
    background: rgba(244,163,64,0.10);
}

#bird-header h1 {
    position: relative;
    z-index: 2;

    color: white;
    margin: 0 0 10px 0;

    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

#bird-header p {
    position: relative;
    z-index: 2;

    color: rgba(255,255,255,0.92);

    margin: 0;
    max-width: 850px;

    font-size: 1rem;
    line-height: 1.7;
}

#bird-header strong {
    color: #fff1c9;
}

.bird-badge {
    position: relative;
    z-index: 2;

    display: inline-block;

    margin-bottom: 15px;
    padding: 8px 14px;

    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.26);

    border-radius: 999px;

    color: white;

    font-size: 0.77rem;
    font-weight: 700;
    letter-spacing: 0.09em;
}

/* INDIA ACCENT */
.india-line {
    position: relative;
    z-index: 2;

    display: flex;
    width: 90px;
    height: 4px;

    margin-top: 18px;

    border-radius: 999px;
    overflow: hidden;
}

.india-line span {
    flex: 1;
}

.saffron {
    background: #ff9933;
}

.white {
    background: #ffffff;
}

.green {
    background: #138808;
}


/* CARDS */
#upload-card,
#result-card {
    background:
        rgba(255,255,255,0.92);

    border:
        1px solid var(--border);

    border-radius:
        22px;

    padding:
        21px;

    box-shadow:
        0 12px 30px
        rgba(34,80,44,0.08);
}

#upload-card h3,
#result-card h3 {
    color: var(--text);

    margin-top: 0;
    margin-bottom: 6px;

    font-size: 1.28rem;
}

.section-note {
    color: var(--muted);

    font-size: 0.92rem;
    line-height: 1.6;

    margin-bottom: 14px;
}


/* IMAGE AREA */
#bird-image {
    border-radius:
        18px !important;

    overflow: hidden;

    border:
        1.4px dashed
        #9ac3a2 !important;

    background:
        #f7fcf6 !important;
}


/* BUTTONS */
#classify-btn {
    background:
        linear-gradient(
            90deg,
            #2f6b3f,
            #579363
        ) !important;

    color:
        white !important;

    border:
        none !important;

    border-radius:
        13px !important;

    font-weight:
        700 !important;

    min-height:
        46px !important;
}

#classify-btn:hover {
    background:
        linear-gradient(
            90deg,
            #244f32,
            #467d53
        ) !important;
}

#clear-btn {
    border-radius:
        13px !important;

    min-height:
        46px !important;
}


/* PREDICTIONS */
#prediction-output {
    border-radius:
        16px !important;
}

#result-output textarea {
    border-radius:
        14px !important;

    background:
        #fbfdf9 !important;
}


/* NOTE */
#bird-note {
    margin-top:
        18px;

    padding:
        13px 15px;

    background:
        #f7fbf4;

    border-left:
        4px solid
        var(--saffron);

    border-radius:
        12px;

    color:
        #4d5e50;

    font-size:
        0.87rem;

    line-height:
        1.55;
}


/* FOOTER */
#footer-note {
    text-align:
        center;

    color:
        #708076;

    font-size:
        0.78rem;

    margin-top:
        20px;
}
"""


# =========================================================
# GRADIO APP
# =========================================================

with gr.Blocks(
    css=CUSTOM_CSS,
    title="Indian Birds Species Classifier"
) as demo:

    with gr.Column(
        elem_id="bird-shell"
    ):

        gr.HTML(
            """
            <div id="bird-header">

                <div class="bird-badge">
                    INDIAN AVIFAUNA • RESNET50V2
                </div>

                <h1>
                    🦅 Indian Birds Species Classifier
                </h1>

                <p>
                    Upload an image of an
                    <strong>Indian bird species</strong>
                    and let the AI model identify the most likely species
                    with Top 5 confidence scores.
                </p>

                <div class="india-line">
                    <span class="saffron"></span>
                    <span class="white"></span>
                    <span class="green"></span>
                </div>

            </div>
            """
        )


        with gr.Row(
            equal_height=True
        ):


            # LEFT CARD
            with gr.Column(
                scale=1,
                elem_id="upload-card"
            ):

                gr.HTML(
                    """
                    <h3>Upload Bird Image</h3>

                    <div class="section-note">
                        Choose a clear bird image.
                        Natural-light photos usually provide better predictions.
                    </div>
                    """
                )

                image_input = gr.Image(
                    type="pil",
                    label="Bird Image",
                    elem_id="bird-image"
                )

                with gr.Row():

                    clear_button = gr.Button(
                        "Clear",
                        elem_id="clear-btn"
                    )

                    analyze_button = gr.Button(
                        "Classify Bird",
                        variant="primary",
                        elem_id="classify-btn"
                    )


            # RIGHT CARD
            with gr.Column(
                scale=1,
                elem_id="result-card"
            ):

                gr.HTML(
                    """
                    <h3>Prediction Results</h3>

                    <div class="section-note">
                        The model displays the five most likely
                        Indian bird species.
                    </div>
                    """
                )

                prediction_output = gr.Label(
                    num_top_classes=5,
                    label="Top 5 Predictions",
                    elem_id="prediction-output"
                )

                result_output = gr.Textbox(
                    label="Best Match",
                    lines=4,
                    elem_id="result-output"
                )

                gr.HTML(
                    """
                    <div id="bird-note">
                        <strong>Note:</strong>
                        Prediction quality depends on image clarity,
                        angle and bird visibility.
                    </div>
                    """
                )


        gr.HTML(
            """
            <div id="footer-note">
                Indian Birds Species Classifier
                • ResNet50V2
                • TensorFlow / Keras
                • Gradio
            </div>
            """
        )


    # EVENTS

    analyze_button.click(
        fn=predict_bird,
        inputs=image_input,
        outputs=[
            prediction_output,
            result_output
        ]
    )

    clear_button.click(
        fn=clear_app,
        inputs=None,
        outputs=[
            image_input,
            prediction_output,
            result_output
        ]
    )


if __name__ == "__main__":
    demo.launch()
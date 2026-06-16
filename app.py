import os
import numpy as np
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import io
from PIL import Image

app = Flask(__name__)

# =====================================================
# CONFIGURATION
# =====================================================

IMG_SIZE    = (224, 224)

# Your remapped label encoding:
# 0 = Clean, 1 = Moderate, 2 = Dirty
CLASS_NAMES = ["Clean", "Moderate", "Dirty"]

# Ordinal ranks for NEATNET score
# Clean=3 (best), Moderate=2, Dirty=1 (worst)
CLASS_RANKS = [3, 2, 1]

def get_tag(score):
    if score >= 66:
        return "Clean"
    elif score >= 34:
        return "Moderate"
    else:
        return "Dirty"

def get_note(tag, score, probs):
    clean_pct    = round(probs[0] * 100, 1)
    moderate_pct = round(probs[1] * 100, 1)
    dirty_pct    = round(probs[2] * 100, 1)
    return (
        f"Clean {clean_pct}%  •  "
        f"Moderate {moderate_pct}%  •  "
        f"Dirty {dirty_pct}%"
    )

# =====================================================
# LOAD MODELS ONCE AT STARTUP
# =====================================================

print("Loading ResNet50 feature extractor...")
resnet_base = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)
feature_extractor = Model(
    inputs=resnet_base.input,
    outputs=resnet_base.output
)

print("Loading XGBoost classifier...")
classifier = xgb.XGBClassifier()
classifier.load_model('best_model.json')

print("Loading scaler...")
scaler = joblib.load('neatnet_scaler.pkl')

print("All models loaded. Ready.")

# =====================================================
# ROUTES
# =====================================================

@app.route('/')
def index():
    return render_template('neatnet_v4.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # ── 1. Load and preprocess image ─────────────────
        img = Image.open(
            io.BytesIO(file.read())
        ).convert('RGB')

        img       = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        # ── 2. ResNet50 feature extraction ───────────────
        x        = preprocess_input(img_array)
        features = feature_extractor.predict(
            x, verbose=0
        )                                               # (1, 2048)

        # ── 3. Normalize with saved scaler ───────────────
        features = scaler.transform(features)           # (1, 2048)

        # ── 4. XGBoost predict ────────────────────────────
        # XGBoost uses flat 2048 features directly
        # NO reshape needed unlike CNN+BiLSTM
        probs = classifier.predict_proba(features)[0].tolist()  # (3,)

        # ── 5. Compute NEATNET score ───────────────────────
        expected_rank = sum(
            p * r for p, r in zip(probs, CLASS_RANKS)
        )
        neatnet_score = ((expected_rank - 1) / 2) * 100
        neatnet_score = float(np.clip(neatnet_score, 0, 100))

        # ── 6. Determine tag and note ─────────────────────
        tag  = get_tag(neatnet_score)
        note = get_note(tag, neatnet_score, probs)

        # ── 7. Return response ────────────────────────────
        return jsonify({
    'score'    : round(neatnet_score, 2),
    'tag'      : tag,
    'note'     : note,
    'prob_clean'    : round(probs[0] * 100, 1),
    'prob_moderate' : round(probs[1] * 100, 1),
    'prob_dirty'    : round(probs[2] * 100, 1)
})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)

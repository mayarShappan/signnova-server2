# ─────────────────────────────────────────
#  SignNova — Flask API Server (Words)
#  يشغل مودل الكلمات ويقبل frames من الويبسايت
#
#  تشغيل:
#    pip install flask flask-cors mediapipe --break-system-packages
#    python server_words.py
#
#  بعدين افتح الويبسايت → Demo Words وهيتوصل أوتوماتيك
# ─────────────────────────────────────────

import os
import sys
import pickle
import base64
from itertools import combinations

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

import mediapipe as mp

# ─────────────────────────────────────────
#  Paths  (نفس folder الـ server_words.py)
# ─────────────────────────────────────────
MODEL_PATH   = "sign_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
CONFIG_PATH  = "model_config.pkl"

for f in [MODEL_PATH, ENCODER_PATH, CONFIG_PATH]:
    if not os.path.exists(f):
        print(f"[ERROR] ملف مش موجود: {f}")
        print("        تأكد إن الـ server شغال في نفس folder الـ .pkl files")
        sys.exit(1)

# ─────────────────────────────────────────
#  Load Model
# ─────────────────────────────────────────
print("⏳ Loading word model...")
with open(MODEL_PATH,   "rb") as f: model         = pickle.load(f)
with open(ENCODER_PATH, "rb") as f: label_encoder = pickle.load(f)
with open(CONFIG_PATH,  "rb") as f: config        = pickle.load(f)

HAND_DIM         = config["hand_dim"]
POSE_DIM         = config["pose_dim"]
TOTAL_FRAME_DIM  = config["total_frame_dim"]
FEATURE_DIM      = config["feature_dim"]
FRAMES_PER_VIDEO = config["frames_per_video"]
HAND_DISTS_DIM   = config["hand_dist_dim"]
HAND_DIST_SINGLE = HAND_DISTS_DIM // 2

CLASSES = list(label_encoder.classes_)
print(f"✅ Model loaded! Classes ({len(CLASSES)}): {CLASSES}")

# ─────────────────────────────────────────
#  MediaPipe Holistic
# ─────────────────────────────────────────
print("⏳ Loading MediaPipe Holistic...")
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ MediaPipe ready!")

# ─────────────────────────────────────────
#  Feature Extraction  (نفس camera.py)
# ─────────────────────────────────────────
HAND_PAIRS = list(combinations(range(21), 2))

def compute_hand_distances(hand_flat):
    arr = np.array(hand_flat, dtype=np.float32)
    if np.all(arr == 0):
        return np.zeros(HAND_DIST_SINGLE, dtype=np.float32)
    pts   = arr.reshape(21, 3)
    dists = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in HAND_PAIRS], dtype=np.float32)
    return dists / (dists.max() + 1e-6)

def make_relative(landmarks_flat, ref_idx=0):
    arr = np.array(landmarks_flat, dtype=np.float32)
    if np.all(arr == 0):
        return arr.tolist()
    pts      = arr.reshape(-1, 3)
    pts     -= pts[ref_idx]
    max_dist = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts     /= max_dist
    return pts.flatten().tolist()

def get_zeros(dim):
    return [0.0] * dim

def extract_frame_vector_from_result(result):
    """استخرج feature vector من نتيجة MediaPipe Holistic"""
    if result.left_hand_landmarks:
        raw      = [v for lm in result.left_hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
        lh       = make_relative(raw)
        lh_dists = compute_hand_distances(lh)
    else:
        lh       = get_zeros(HAND_DIM)
        lh_dists = np.zeros(HAND_DIST_SINGLE, dtype=np.float32)

    if result.right_hand_landmarks:
        raw      = [v for lm in result.right_hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
        rh       = make_relative(raw)
        rh_dists = compute_hand_distances(rh)
    else:
        rh       = get_zeros(HAND_DIM)
        rh_dists = np.zeros(HAND_DIST_SINGLE, dtype=np.float32)

    if result.pose_landmarks:
        raw  = [v for lm in result.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
        pose = make_relative(raw, ref_idx=0)
    else:
        pose = get_zeros(POSE_DIM)

    return lh + rh + pose + lh_dists.tolist() + rh_dists.tolist()

def aggregate(frames):
    arr      = np.array(frames, dtype=np.float32)
    mean     = arr.mean(axis=0)
    std      = arr.std(axis=0)
    if len(frames) >= 2:
        diff     = np.diff(arr, axis=0)
        vel_mean = diff.mean(axis=0)
        vel_std  = diff.std(axis=0)
    else:
        vel_mean = np.zeros(TOTAL_FRAME_DIM, dtype=np.float32)
        vel_std  = np.zeros(TOTAL_FRAME_DIM, dtype=np.float32)
    return np.concatenate([mean, std, vel_mean, vel_std])

def resample_to_fixed(frames_list, target=FRAMES_PER_VIDEO):
    n   = len(frames_list)
    arr = np.array(frames_list, dtype=np.float32)
    if n == target:
        return arr
    indices = np.linspace(0, n - 1, target)
    lo      = np.floor(indices).astype(int)
    hi      = np.minimum(lo + 1, n - 1)
    frac    = (indices - lo)[:, np.newaxis]
    return arr[lo] * (1 - frac) + arr[hi] * frac

def predict_from_frames(frames_list):
    if len(frames_list) < 3:
        return None, 0.0, []

    resampled = resample_to_fixed(frames_list)
    vec       = aggregate(resampled.tolist())

    if vec is None or len(vec) != FEATURE_DIM:
        return None, 0.0, []

    proba    = model.predict_proba(vec.reshape(1, -1))[0]
    top3     = sorted(zip(CLASSES, proba), key=lambda x: -x[1])[:3]

    class_idx  = int(np.argmax(proba))
    confidence = float(proba[class_idx])
    label      = CLASSES[class_idx].rstrip("0123456789")

    top3_out = [{"word": w.rstrip("0123456789"), "confidence": float(c)} for w, c in top3]
    return label, confidence, top3_out

# ─────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/health_words", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": len(CLASSES), "model": "WordModel_pkl"})

@app.route("/predict_words", methods=["POST"])
def predict_words():
    """
    يستقبل frames (list of base64 images) من الويبسايت،
    يستخرج Holistic landmarks من كل frame،
    يرجع predicted word + confidence + top3
    """
    try:
        import cv2
        data   = request.get_json(force=True)
        frames_b64 = data.get("frames")  # list of base64 strings

        if not frames_b64 or len(frames_b64) < 3:
            return jsonify({"error": "محتاج على الأقل 3 frames"}), 400

        frames_vectors = []

        for img_b64 in frames_b64:
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]

            img_bytes = base64.b64decode(img_b64)
            img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(rgb)

            has_hand = (result.left_hand_landmarks is not None or
                        result.right_hand_landmarks is not None)

            if has_hand or result.pose_landmarks:
                vec = extract_frame_vector_from_result(result)
                frames_vectors.append(vec)

        if len(frames_vectors) < 3:
            return jsonify({
                "hand_detected": False,
                "word":          None,
                "confidence":    0,
                "top3":          [],
                "frames_used":   len(frames_vectors),
            })

        label, confidence, top3 = predict_from_frames(frames_vectors)

        return jsonify({
            "hand_detected": True,
            "word":          label,
            "confidence":    confidence,
            "top3":          top3,
            "frames_used":   len(frames_vectors),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/predict_words_landmarks", methods=["POST"])
def predict_words_landmarks():
    """
    بديل أسرع: يستقبل frames كـ landmarks مباشرة
    كل frame = list of 63 floats (21 نقطة × 3 للـ right hand)
    أو dict فيه left_hand, right_hand, pose
    """
    try:
        data   = request.get_json(force=True)
        frames = data.get("frames")  # list of frame vectors

        if not frames or len(frames) < 3:
            return jsonify({"error": "محتاج على الأقل 3 frames"}), 400

        frames_vectors = [np.array(f, dtype=np.float32) for f in frames]

        label, confidence, top3 = predict_from_frames(frames_vectors)

        return jsonify({
            "hand_detected": True,
            "word":          label,
            "confidence":    confidence,
            "top3":          top3,
            "frames_used":   len(frames_vectors),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SignNova Words API Server")
    print("  http://localhost:5001")
    print("=" * 55)
    print("  افتح الويبسايت → Demo Words وهيتوصل بالموديل")
    print("  اضغط Ctrl+C عشان توقف الـ server\n")
    app.run(host="0.0.0.0", port=5001, debug=False)

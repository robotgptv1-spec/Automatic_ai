import os
import traceback
from flask import Flask, request, jsonify, render_template, send_file, g
import zipfile as zp
from ml_engine import AutoMLSession
from cnn_task import AutoImageSession
from ai_autoconfig import suggest_config, AIConfigError
from firebase_auth import init_firebase, require_auth

app = Flask(__name__)
init_firebase()
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024 
# Is line ko add karein taaki Flask heavy incoming content streams ko direct reject na kare
app.config['MAX_FORM_MEMORY_SIZE'] = 100 * 1024 * 1024
app.config['PROPAGATE_EXCEPTIONS'] = True
# In-memory session store: session_id -> AutoMLSession / AutoImageSession
SESSIONS = {}


def get_session(session_id):
    session = SESSIONS.get(session_id)
    if session is None:
        raise KeyError("Unknown or expired session_id. Upload a dataset again.")
    # If Firebase auth is configured, make sure this session belongs to the
    # caller — otherwise one signed-in user could poke at another user's
    # session_id (session ids are just short hex strings, not secrets).
    owner = getattr(session, "owner_uid", None)
    if owner is not None and owner != getattr(g, "uid", None):
        raise KeyError("Unknown or expired session_id. Upload a dataset again.")
    return session


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
@require_auth
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["file"]
    
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".zip")):
        return jsonify({"error": "Only .csv and .zip files are supported right now."}), 400

    # Task type ke hisaab se session toggle karein
    if filename.endswith(".zip"):
        session = AutoImageSession()
        try:
            summary = session.load_image_zip(file)
        except Exception as e:
            return jsonify({"error": f"Could not read ZIP: {e}"}), 400
    else:
        session = AutoMLSession()
        try:
            summary = session.load_csv(file)
        except Exception as e:
            return jsonify({"error": f"Could not read CSV: {e}"}), 400

    session.owner_uid = g.uid
    SESSIONS[session.id] = session
    return jsonify(summary)


@app.route("/api/configure", methods=["POST"])
@require_auth
def configure():
    data = request.get_json(force=True)
    try:
        session = get_session(data.get("session_id"))
        
        # Image classification ke liye explicit configuration ki zaroorat nahi hai
        if session.task_type == "image_classification":
            return jsonify({
                "task_type": session.task_type,
                "categorical_features": [],
                "numeric_features": []
            })
            
        result = session.configure(
            feature_columns=data.get("feature_columns", []),
            target_column=data.get("target_column"),
            task_type=data.get("task_type", "auto"),
        )
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


@app.route("/api/ai_autoconfigure", methods=["POST"])
@require_auth
def ai_autoconfigure():
    """Uses a user-supplied LLM (OpenAI / Anthropic / Gemini) key + a plain
    text prompt to pick the target column, feature columns, and task type
    automatically, then applies that config to the session (same effect as
    /api/configure, just decided by the model instead of the UI)."""
    data = request.get_json(force=True)
    try:
        session = get_session(data.get("session_id"))
        if session.task_type == "image_classification":
            return jsonify({"error": "AI auto-configure is for tabular (CSV) datasets only — image datasets already train automatically."}), 400

        provider = (data.get("provider") or "").strip().lower()
        api_key = (data.get("api_key") or "").strip()
        prompt = (data.get("prompt") or "").strip()
        if provider not in ("openai", "anthropic", "gemini"):
            return jsonify({"error": "provider must be one of: openai, anthropic, gemini"}), 400
        if not api_key:
            return jsonify({"error": "API key is required."}), 400
        if not prompt:
            return jsonify({"error": "Describe what you want the model to predict."}), 400

        columns_info = session.summary()["columns"]
        suggestion = suggest_config(provider, api_key, prompt, columns_info)

        target_column = suggestion.get("target_column")
        feature_columns = suggestion.get("feature_columns") or []
        task_type = suggestion.get("task_type", "auto")
        reasoning = suggestion.get("reasoning", "")

        valid_cols = set(session.df.columns)
        if target_column not in valid_cols:
            raise ValueError(f"Model picked an unknown target column: {target_column!r}")
        feature_columns = [c for c in feature_columns if c in valid_cols and c != target_column]
        if not feature_columns:
            feature_columns = [c for c in session.df.columns if c != target_column]

        result = session.configure(
            feature_columns=feature_columns,
            target_column=target_column,
            task_type=task_type if task_type in ("classification", "regression") else "auto",
        )
        result.update({
            "target_column": target_column,
            "feature_columns": feature_columns,
            "reasoning": reasoning,
        })
        return jsonify(result)
    except AIConfigError as e:
        return jsonify({"error": str(e)}), 502
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"AI auto-configure failed: {e}"}), 500


@app.route("/api/train", methods=["POST"])
@require_auth
def train():
    data = request.get_json(force=True)
    try:
        session = get_session(data.get("session_id"))
        epochs = int(data.get("epochs", 30))
        lr = float(data.get("lr", 0.001))
        batch_size = int(data.get("batch_size", 32))
        test_size = float(data.get("test_size", 0.2))
        
        epochs = max(1, min(epochs, 300))
        batch_size = max(1, min(batch_size, 512))
        test_size = min(max(test_size, 0.05), 0.5)

        result = session.train(test_size=test_size, epochs=epochs, lr=lr, batch_size=batch_size)
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Training failed: {e}"}), 500

    # Frontend compatibility ke liye conditional JSON response
    if session.task_type == "image_classification":
        return jsonify({
            "log": result["log"],
            "final": result["final"],
            "problem_mode": result["problem_mode"],
            "class_names": session.class_names,
            "feature_columns": [],
            "categorical_features": [],
            "categories": {}
        })

    return jsonify({
        "log": result["log"],
        "final": result["final"],
        "problem_mode": result["problem_mode"],
        "class_names": session.class_names,
        "feature_columns": session.feature_columns,
        "categorical_features": session.categorical_features,
        "categories": {c: list(session.cat_encoders[c].classes_) for c in session.categorical_features},
    })


@app.route("/api/predict", methods=["POST"])
@require_auth
def predict():
    # 1. Image prediction handle karein (Multipart Form File Upload)
    if "image" in request.files:
        try:
            file = request.files["image"]
            session_id = request.form.get("session_id")
            session = get_session(session_id)
            
            if session.model is None:
                return jsonify({"error": "Train an image model before predicting."}), 400
                
            result = session.predict_one(file.read())
            return jsonify(result)
        except (KeyError, ValueError) as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {e}"}), 500

    # 2. Existing Tabular prediction logic (JSON Data)
    data = request.get_json(force=True)
    try:
        session = get_session(data.get("session_id"))
        if session.model is None:
            return jsonify({"error": "Train a model before predicting."}), 400
        result = session.predict_one(data.get("features", {}))
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500
    return jsonify(result)


@app.route("/api/download_model/<session_id>", methods=["GET"])
@require_auth
def download_model(session_id):
    try:
        session = get_session(session_id)
        if session.model is None:
            return jsonify({"error": "Train a model before downloading."}), 400
    except KeyError as e:
        return jsonify({"error": str(e)}), 404
        
    buf = session.to_bytes()
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"autoai_model_{session_id}.pkl",
        mimetype="application/octet-stream",
    )


@app.route("/api/reset", methods=["POST"])
@require_auth
def reset():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    try:
        get_session(session_id)  # raises KeyError if missing or not owned by this user
    except KeyError:
        return jsonify({"ok": True})  # nothing to do, already gone / not yours
    SESSIONS.pop(session_id, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

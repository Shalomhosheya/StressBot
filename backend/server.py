"""
server.py  —  Flask API for the Serenity local stress-support chatbot.

Endpoints
  POST /chat    { message, session_id? }  →  { response, intent, session_id }
  POST /reset   { session_id }            →  { status }
  GET  /health                            →  { status, model }
"""

import uuid
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatmodel import generate_response, classify_intent

app = Flask(__name__)
CORS(app)

# session_id → chat_history_ids tensor (or None)
sessions: dict[str, torch.Tensor | None] = {}


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json(force=True)
    user_input = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    history = sessions.get(session_id)   # tensor or None
    intent  = classify_intent(user_input)

    response, updated_history = generate_response(user_input, history)
    sessions[session_id] = updated_history

    return jsonify({
        "response":   response,
        "intent":     intent,
        "session_id": session_id,
    })


@app.route("/reset", methods=["POST"])
def reset():
    data       = request.get_json(force=True)
    session_id = data.get("session_id", "")
    sessions.pop(session_id, None)
    return jsonify({"status": "reset", "session_id": session_id})


if __name__ == "__main__":
    print("Serenity server starting on http://localhost:5000")
    app.run(debug=False, port=5000, threaded=False)  # threaded=False → single worker (model is not thread-safe)
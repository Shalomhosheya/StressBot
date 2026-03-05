from flask import Flask, request, jsonify
from flask_cors import CORS
from chatmodel import generate_response      # ← change to chatmodel if your file has no underscore

app = Flask(__name__)
CORS(app)

chat_history_ids = None

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids
    data = request.json
    user_input = data.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    response, chat_history_ids = generate_response(user_input, chat_history_ids)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
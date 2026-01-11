import os
import argparse
from flask import Flask, request, jsonify
from litellm import completion
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)


@app.route('/chat', methods=['POST'])
def chat():
    # Check authentication
    server_key = app.config.get('GLEE_KEY')
    if server_key:
        request_key = request.headers.get('X-API-Key')
        if request_key != server_key:
            return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    messages = data.get('messages', [])
    game_params = data.get('game_params', {})
    decision_needed = data.get('decision', False)
    
    print(f"Received request for {game_params.get('public_name', 'Unknown Player')}")
    
    # You can use game_params to adjust the system prompt or logic if needed
    # For now, we just pass the messages to the LLM
    
    try:
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.llmod.ai/v1",
            model="NEW_MODEL_NAME!!!"
        )
        response_text = llm.invoke(messages).content
        # response_text = response.choices[0].message.content
        print(f"LLM Response: {response_text}")
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    # Optional: Check authentication if header is present
    server_key = app.config.get('GLEE_KEY')
    if server_key:
        request_key = request.headers.get('X-API-Key')
        if request_key and request_key != server_key:
            return jsonify({"status": "alive", "auth": "failed"}), 401
        elif request_key == server_key:
            return jsonify({"status": "alive", "auth": "ok"}), 200
            
    return jsonify({"status": "alive"}), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--key', type=str, help='API Key for authentication')
    args = parser.parse_args()
    
    # Set API key in app config
    app.config['GLEE_KEY'] = args.key or os.getenv("GLEE_KEY")
    
    if not app.config['GLEE_KEY']:
        print("WARNING: No API Key configured! The server is running in insecure mode.")
    
    app.run(host=args.host, port=args.port, debug=True)

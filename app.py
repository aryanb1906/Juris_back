from flask import Flask, request, jsonify
from flask_cors import CORS

from src.config import Config
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.response_generator import ResponseGenerator

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# Initialize and load your AI components once for efficiency.
vector_store = VectorStore(Config.EMBEDDING_MODEL)
vector_store.load_data(Config.CSV_PATH)
query_engine = QueryEngine(vector_store)
response_generator = ResponseGenerator(Config.GEMINI_API_KEY, Config.MODEL_NAME)

@app.route('/api/legal-advice', methods=['POST'])
def legal_advice():
    # Get JSON data from frontend.
    data = request.get_json()
    query = data.get("text")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Use your existing logic to process the query.
    relevant_sections = query_engine.process_query(query)
    legal_response = response_generator.generate_response(query, relevant_sections)
    
    # Return the generated legal advice as JSON.
    return jsonify({"response": legal_response}), 200

@app.route('/')
def home():
    return "Juris AI Legal Advisor API is running."

if __name__ == '__main__':
    app.run(debug=True)

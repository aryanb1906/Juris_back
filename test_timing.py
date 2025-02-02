import time
from src.config import Config
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.response_generator import ResponseGenerator

# Initialize all components (reuse logic from your main program)
vector_store = VectorStore(Config.EMBEDDING_MODEL)
vector_store.load_data(Config.CSV_PATH)
query_engine = QueryEngine(vector_store)
response_generator = ResponseGenerator(Config.GEMINI_API_KEY, Config.MODEL_NAME)

def measure_response_time(query):
    start_time = time.time()
    relevant_sections = query_engine.process_query(query)
    legal_response = response_generator.generate_response(query, relevant_sections)
    end_time = time.time()
    return legal_response, end_time - start_time

if __name__ == "__main__":
    for i in range(5):
        test_query = "What are the legal implications of theft under IPC?"
        response, elapsed = measure_response_time(test_query)
        print("Legal Response:\n", response)
        print(f"Time taken: {elapsed:.2f} seconds")
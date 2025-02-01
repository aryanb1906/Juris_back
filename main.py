from src.config import Config
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.response_generator import ResponseGenerator

def main():
    # Initialize components
    vector_store = VectorStore(Config.EMBEDDING_MODEL)
    vector_store.load_data(Config.CSV_PATH)
    
    query_engine = QueryEngine(vector_store)
    response_generator = ResponseGenerator(
        Config.GEMINI_API_KEY,
        Config.MODEL_NAME
    )
    
    print("Juris AI Legal Oracle - Ready to assist")
    print("======================================")
    
    while True:
        query = input("\nEnter your legal query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        relevant_sections = query_engine.process_query(query)
        response = response_generator.generate_response(query, relevant_sections)
        
        print("\nLegal Analysis:")
        print("=" * 80)
        print(response)
        print("=" * 80)

if __name__ == "__main__":
    main()
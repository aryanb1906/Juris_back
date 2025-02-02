from src.config import Config
from src.vector_store import VectorStore
from src.query_engine import QueryEngine
from src.response_generator import ResponseGenerator

# Sample test queries and their expected results.
# In a real scenario, these would be prepared by a domain expert.
test_data = [
    {
        "query": "What does the law say about murder?",
        "expected": {"IPC_302", "IPC_304"}
    },
    {
        "query": "Explain the laws regarding theft.",
        "expected": {"IPC_378"}
    }
]

# Initialize components
vector_store = VectorStore(Config.EMBEDDING_MODEL)
vector_store.load_data(Config.CSV_PATH)
query_engine = QueryEngine(vector_store)

def evaluate_retrieval(test_data):
    all_precisions = []
    all_recalls = []
    for item in test_data:
        query = item["query"]
        expected = item["expected"]
        relevant_sections = query_engine.process_query(query)
        # Convert the fetched sections to a set of their identifiers.
        retrieved = set(relevant_sections['Section'])
        
        # Calculate precision and recall.
        true_positives = len(retrieved.intersection(expected))
        precision = true_positives / len(retrieved) if retrieved else 0
        recall = true_positives / len(expected) if expected else 0
        all_precisions.append(precision)
        all_recalls.append(recall)
        print(f"Query: {query}")
        print("Retrieved Sections:", retrieved)
        print("Expected Sections:", expected)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}\n")
    
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")

if __name__ == "__main__":
    for i in range(3):
        print(f"Iteration {i+1}")
        evaluate_retrieval(test_data)
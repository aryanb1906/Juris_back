# generate_cache.py
from src.config import Config
from src.vector_store import VectorStore

if __name__ == "__main__":
    vs = VectorStore(Config.EMBEDDING_MODEL)
    vs.load_data(Config.CSV_PATH)
    print("Cache generated successfully at data/ipc_cache.pkl")

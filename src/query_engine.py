from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QueryEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def process_query(self, query):
        query_embedding = self.vector_store.model.encode([query])
        similarities = cosine_similarity(
            query_embedding,
            self.vector_store.embeddings
        )[0]
        
        top_k = 5
        relevant_indices = np.argsort(similarities)[-top_k:][::-1]
        return self.vector_store.df.iloc[relevant_indices][
            ['Section', 'Offense', 'Description', 'Punishment']
        ]
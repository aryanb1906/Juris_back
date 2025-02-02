# from sentence_transformers import SentenceTransformer
# import pandas as pd
# import numpy as np

# class VectorStore:
#     def __init__(self, model_name):
#         self.model = SentenceTransformer(model_name)
#         self.df = None
#         self.embeddings = None
    
#     def load_data(self, csv_path):
#         self.df = pd.read_csv(csv_path)
#         self.df['Offense'] = self.df['Offense'].fillna('Not specified')
#         self.df['Punishment'] = self.df['Punishment'].fillna('Not specified')
        
#         self.df['combined_text'] = self.df.apply(
#             lambda x: f"""Section {x['Section']}:
#             Offense: {x['Offense']}
#             Description: {x['Description']}
#             Punishment: {x['Punishment']}""".strip(),
#             axis=1
#         )
        
#         return self.generate_embeddings()
    
#     def generate_embeddings(self):
#         self.embeddings = self.model.encode(self.df['combined_text'].tolist())
#         return self.embeddings

# vector_store.py
import os
import pickle
from datetime import datetime
from sentence_transformers import SentenceTransformer
import pandas as pd

class VectorStore:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.embeddings = None
        self.cache_path = "data/ipc_cache.pkl"

    def load_data(self, csv_path):
        # Check if cache exists and is up-to-date
        if self._cache_valid(csv_path):
            self._load_cache()
        else:
            self._process_csv(csv_path)
            self._save_cache()
        return self.df

    def _cache_valid(self, csv_path):
        """Check if cache exists and is newer than CSV file"""
        if not os.path.exists(self.cache_path):
            return False
        
        cache_time = os.path.getmtime(self.cache_path)
        csv_time = os.path.getmtime(csv_path)
        return cache_time > csv_time

    def _process_csv(self, csv_path):
        """Process CSV and generate embeddings"""
        self.df = pd.read_csv(csv_path)
        self.df['Offense'] = self.df['Offense'].fillna('Not specified')
        self.df['Punishment'] = self.df['Punishment'].fillna('Not specified')
        self.df['combined_text'] = self.df.apply(
            lambda x: f"Section {x['Section']}:\n{x['Offense']}\n{x['Description']}\n{x['Punishment']}", 
            axis=1
        )
        self.embeddings = self.model.encode(self.df['combined_text'].tolist())

    def _save_cache(self):
        """Save processed data to pickle file"""
        with open(self.cache_path, 'wb') as f:
            pickle.dump((self.df, self.embeddings), f)

    def _load_cache(self):
        """Load data from pickle cache"""
        with open(self.cache_path, 'rb') as f:
            self.df, self.embeddings = pickle.load(f)

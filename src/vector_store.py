from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

class VectorStore:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.embeddings = None
    
    def load_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['Offense'] = self.df['Offense'].fillna('Not specified')
        self.df['Punishment'] = self.df['Punishment'].fillna('Not specified')
        
        self.df['combined_text'] = self.df.apply(
            lambda x: f"""Section {x['Section']}:
            Offense: {x['Offense']}
            Description: {x['Description']}
            Punishment: {x['Punishment']}""".strip(),
            axis=1
        )
        
        return self.generate_embeddings()
    
    def generate_embeddings(self):
        self.embeddings = self.model.encode(self.df['combined_text'].tolist())
        return self.embeddings
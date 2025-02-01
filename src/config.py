import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    MODEL_NAME = 'models/gemini-pro'
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    CSV_PATH = 'data/ipc_sections.csv'
    
    COLUMNS = {
        'SECTION': 'Section',
        'OFFENSE': 'Offense',
        'DESCRIPTION': 'Description',
        'PUNISHMENT': 'Punishment'
    }
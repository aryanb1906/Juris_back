import google.generativeai as genai

class ResponseGenerator:
    def __init__(self, api_key, model_name):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate_response(self, query, relevant_sections):
        prompt = f"""
        Legal Query: {query}
        
        Relevant IPC Sections:
        
        {relevant_sections.apply(lambda x: f"""
        Section: {x['Section']}
        Offense: {x['Offense']}
        Description: {x['Description']}
        Punishment: {x['Punishment']}
        """, axis=1).str.cat(sep='\n')}
        
        Provide a comprehensive legal response including:
        1. Analysis of applicable IPC sections
        2. Legal implications and consequences
        3. Recommended actions
        4. Rights under Indian law
        """
        
        response = self.model.generate_content(prompt)
        return response.text
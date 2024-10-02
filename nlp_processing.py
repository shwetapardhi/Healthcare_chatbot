# nlp_processing.py
import spacy

class NLPProcessor:
    def __init__(self):
        # Load a small English model for entity extraction
        self.nlp = spacy.load("en_core_web_sm")

    def extract_symptoms(self, user_input: str):
        doc = self.nlp(user_input.lower())
        symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]
        return ', '.join(symptoms) if symptoms else user_input

# Instantiate the NLP Processor
nlp_processor = NLPProcessor()

# chatbot_model.py
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

class SymptomCheckerModel:
    def __init__(self):
        # Example dataset of symptoms and conditions
        data = {
            'symptoms': [
                'cough, fever, fatigue',
                'headache, dizziness',
                'sore throat, cough, fever',
                'chest pain, shortness of breath',
                'fatigue, weight loss, nausea'
            ],
            'condition': [
                'Common Cold',
                'Migraine',
                'Flu',
                'Heart Disease',
                'Diabetes'
            ]
        }

        # Create a DataFrame
        self.df = pd.DataFrame(data)

        # Initialize a basic classifier with CountVectorizer and Naive Bayes
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])

        # Train the model
        self.model.fit(self.df['symptoms'], self.df['condition'])

    def predict_condition(self, symptoms: str):
        return self.model.predict([symptoms])[0]

# Instantiate the model
symptom_checker = SymptomCheckerModel()
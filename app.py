# app.py
from flask import Flask, request, jsonify, render_template
from chatbot_model import symptom_checker
from nlp_processing import nlp_processor

app = Flask(__name__)

# Route for rendering the main page
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint to process user inputs
@app.route("/get_diagnosis", methods=["POST"])
def get_diagnosis():
    user_input = request.json.get("message", "")
    
    # Step 1: Extract symptoms using NLP
    symptoms = nlp_processor.extract_symptoms(user_input)

    # Step 2: Get a condition prediction from the model
    predicted_condition = symptom_checker.predict_condition(symptoms)

    # Step 3: Return the result as JSON
    return jsonify({
        "input_symptoms": symptoms,
        "predicted_condition": predicted_condition
    })

if __name__ == "__main__":
    app.run(debug=True)

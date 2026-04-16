from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import ast

app = Flask(__name__)

print("⏳ Loading Advanced Severity-Weighted Core...")
# Load the machine learning model and dictionaries
model_data = joblib.load('model.pkl')
model = model_data['model']
le = model_data['label_encoder']
master_symptoms = model_data['symptoms_list'] # List of 132 symptom names

# Create user-friendly names for the dropdown (e.g. "mild_fever" -> "Mild Fever")
symptoms_dropdown = [{"id": sym, "name": sym.replace('_', ' ').title()} for sym in master_symptoms]

@app.route('/')
def home():
    # Render the HTML frontend and pass the symptoms list to it
    return render_template('index.html', symptoms_list=symptoms_dropdown)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_symptoms = data.get('symptoms', []) # Example: ['itching', 'mild_fever']

    if not user_symptoms:
        return jsonify({'error': 'Please select at least one symptom.'})

    # 1. Create the binary vector (132 zeros)
    input_vector = np.zeros(len(master_symptoms))
    
    # 2. Get the severity dictionary we saved in the model
    severity_dict = model_data.get('severity_dict', {})
    
    # 3. Flip the switches for selected symptoms using actual Medical Severity scores
    for sym in user_symptoms:
        if sym in master_symptoms:
            idx = master_symptoms.index(sym)
            # Apply the weight! (Default to 1 if it's not in the CSV)
            input_vector[idx] = severity_dict.get(sym, 1)

    # 4. Predict probabilities for all diseases
    probs = model.predict_proba([input_vector])[0]
    
    # 5. Get the Top 3 highest probabilities
    top3_indices = np.argsort(probs)[-3:][::-1] 
    
    top3_diseases = []
    for idx in top3_indices:
        disease_name = le.inverse_transform([idx])[0]
        confidence = probs[idx] * 100
        if confidence > 0:
            top3_diseases.append({'name': disease_name.title(), 'confidence': round(confidence, 1)})

    if not top3_diseases:
        return jsonify({'error': 'Could not diagnose. Please add more symptoms.'})

    # 6. Extract data for the primary diagnosis
    primary_disease = top3_diseases[0]['name'].lower()
    
    # Clean up the medications string safely
    meds_raw = model_data['medications'].get(primary_disease, "Consult a doctor.")
    try: 
        meds_clean = ", ".join(ast.literal_eval(meds_raw))
    except: 
        meds_clean = meds_raw

    # 7. Send the complete package back to the frontend
    return jsonify({
        'primary_disease': top3_diseases[0]['name'],
        'primary_confidence': top3_diseases[0]['confidence'],
        'alternatives': top3_diseases[1:], 
        'description': model_data['descriptions'].get(primary_disease, "N/A"),
        'medications': meds_clean,
        'precautions': model_data['precautions'].get(primary_disease, []),
        'diet': model_data['diets'].get(primary_disease, "Balanced diet."),
        'workout': model_data['workouts'].get(primary_disease, "Rest.")
    })

if __name__ == '__main__':
    app.run(debug=True)
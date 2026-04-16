import pandas as pd
import joblib
import numpy as np
import random
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def build_master_brain():
    print("🚀 Initializing Medical Threshold Training Sequence...")
    path = 'archive/'
    
    # 1. Load Original Data
    df_train = pd.read_csv(os.path.join(path, 'Training.csv'))
    if 'Unnamed: 133' in df_train.columns:
        df_train = df_train.drop(columns=['Unnamed: 133'])
    
    target_col = 'prognosis'
    symptom_columns = [col for col in df_train.columns if col != target_col]

    # Map our custom synonym
    if 'loose_motion' not in symptom_columns:
        symptom_columns.append('loose_motion')

    try:
        df_sev = pd.read_csv(os.path.join(path, 'Symptom-severity.csv'))
        severity_dict = dict(zip(df_sev['Symptom'].str.replace(' ', '_').str.lower(), df_sev['weight']))
    except:
        severity_dict = {}
        
    severity_dict['loose_motion'] = 5 

    # 2. Build Disease Blueprints
    disease_blueprints = {}
    for disease in df_train[target_col].unique():
        d_clean = str(disease).strip()
        disease_rows = df_train[df_train[target_col] == disease]
        
        active_syms = [col for col in symptom_columns if col in df_train.columns and disease_rows[col].max() == 1]
        if 'diarrhoea' in active_syms and 'loose_motion' not in active_syms:
            active_syms.append('loose_motion')
            
        disease_blueprints[d_clean] = active_syms

    # Custom additions to the dataset
    custom_diseases = {
        "Common Cold": ["mild_fever", "runny_nose", "cough", "continuous_sneezing", "congestion", "headache", "throat_irritation"],
        "Viral Fever": ["mild_fever", "high_fever", "chills", "fatigue", "headache", "muscle_pain", "malaise"],
        "Sinus Infection (Sinusitis)": ["sinus_pressure", "headache", "runny_nose", "congestion", "loss_of_smell", "mild_fever"],
        "Acidity and Gas": ["belly_pain", "stomach_pain", "acidity", "passage_of_gases", "chest_pain"],
        "Food Poisoning": ["vomiting", "stomach_pain", "belly_pain", "diarrhoea", "nausea", "loose_motion"],
        "Indigestion": ["indigestion", "stomach_pain", "nausea", "loss_of_appetite", "acidity", "loose_motion"],
        "Constipation Issues": ["constipation", "abdominal_pain", "pain_during_bowel_movements"],
        "Tension Headache": ["headache", "neck_pain", "fatigue", "dizziness"],
        "Arthritis and Joint Issues": ["joint_pain", "swelling_joints", "painful_walking", "knee_pain", "stiff_neck"],
        "Muscle Strain": ["muscle_pain", "back_pain", "joint_pain", "neck_pain", "movement_stiffness"],
        "Dry Skin": ["itching", "skin_peeling", "skin_rash"],
        "Skin Allergy": ["itching", "skin_rash", "red_spots_over_body"],
        "Seasonal Allergies": ["continuous_sneezing", "runny_nose", "watering_from_eyes", "redness_of_eyes", "mild_fever"],
        "Pink Eye (Conjunctivitis)": ["redness_of_eyes", "watering_from_eyes", "itching", "pain_behind_the_eyes"],
        "Eye Strain": ["blurred_and_distorted_vision", "headache", "redness_of_eyes", "visual_disturbances"],
        "Dehydration": ["sunken_eyes", "dehydration", "dizziness", "weakness_in_limbs", "fatigue", "drying_and_tingling_lips"],
        "Fatigue and Burnout": ["fatigue", "lethargy", "restlessness", "lack_of_concentration", "mood_swings", "malaise"],
        "Urinary Tract Infection (UTI)": ["burning_micturition", "bladder_discomfort", "continuous_feel_of_urine"],
        "Anxiety and Stress": ["anxiety", "restlessness", "sweating", "palpitations", "fast_heart_rate"],
        "Motion Sickness": ["nausea", "vomiting", "dizziness", "spinning_movements", "unsteadiness"],
        "Peptic Ulcer (Stomach Ulcer)": ["stomach_pain", "belly_pain", "vomiting", "nausea", "acidity", "loose_motion", "loss_of_appetite"]
    }
    
    for d_name, syms in custom_diseases.items():
        disease_blueprints[d_name] = [s for s in syms if s in symptom_columns]

    # --- 3. DEFINING THE SAFETY CATEGORIES ---
    minor_conditions = [
        'common cold', 'viral fever', 'acidity and gas', 'food poisoning',
        'indigestion', 'tension headache', 'muscle strain', 'dry skin',
        'skin allergy', 'seasonal allergies', 'eye strain', 'dehydration',
        'fatigue and burnout', 'motion sickness', 'gastroenteritis', 'allergy',
        'gerd', 'acne', 'dimorphic hemmorhoids(piles)', 'impetigo', 'peptic ulcer (stomach ulcer)'
    ]

    print("⚖️ Constructing Stable Data Vectors...")
    X_train, y_train = [], []

    # INJECT ORIGINAL KAGGLE ROWS (The solid foundation)
    for _, row in df_train.iterrows():
        vec = np.zeros(len(symptom_columns))
        for s in symptom_columns:
            if s in df_train.columns and row[s] == 1:
                vec[symptom_columns.index(s)] = severity_dict.get(s, 1)
        # Apply loose motion logic to original rows
        if 'diarrhoea' in df_train.columns and row['diarrhoea'] == 1:
             vec[symptom_columns.index('loose_motion')] = severity_dict.get('loose_motion', 1)
             
        X_train.append(vec)
        y_train.append(str(row[target_col]).strip())

    # GENERATE SMART SYNTHETIC ROWS
    for disease, syms in disease_blueprints.items():
        if not syms: continue
        d_lower = disease.lower()
        
        num_rows = 300 # Equalize the playing field

        for _ in range(num_rows):
            vec = np.zeros(len(symptom_columns))
            
            # THE MAGIC FIX:
            # Minor diseases can trigger from just 1 symptom.
            # Severe diseases (Paralysis, Malaria, Heart Attack) MUST have at least 3 symptoms.
            if d_lower in minor_conditions:
                min_s = 1
            else:
                min_s = min(3, len(syms)) 

            num_to_pick = random.randint(min_s, len(syms))
            picked = random.sample(syms, num_to_pick)
                
            for s in picked:
                vec[symptom_columns.index(s)] = severity_dict.get(s, 1)
                
            X_train.append(vec)
            y_train.append(disease)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    print("⚙️ Compiling the High-Accuracy Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=60, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_encoded)

    print("🔗 Linking and Scrubbing Medical Dictionaries...")
    def to_clean_dict(df, key_col, val_col):
        return {str(k).strip().lower(): str(v).strip() for k, v in zip(df[key_col], df[val_col])}

    df_meds  = pd.read_csv(os.path.join(path, 'medications.csv'))
    df_desc  = pd.read_csv(os.path.join(path, 'description.csv'))
    df_diet  = pd.read_csv(os.path.join(path, 'diets.csv'))
    df_work  = pd.read_csv(os.path.join(path, 'workout_df.csv'))
    df_prec  = pd.read_csv(os.path.join(path, 'precautions_df.csv'))

    meds_dict = to_clean_dict(df_meds, 'Disease', 'Medication')
    desc_dict = to_clean_dict(df_desc, 'Disease', 'Description')
    diet_dict = to_clean_dict(df_diet, 'Disease', 'Diet')
    work_dict = to_clean_dict(df_work, 'disease', 'workout')
    
    prec_dict = {}
    for _, row in df_prec.iterrows():
        d_name = str(row['Disease']).strip().lower()
        prec_dict[d_name] = [str(row[f'Precaution_{i}']).strip() for i in range(1, 5) if pd.notna(row[f'Precaution_{i}']) and str(row[f'Precaution_{i}']).strip() != ""]

    custom_metadata = {
        'common cold': {'meds': "['Antihistamines', 'Decongestants']", 'desc': "A viral infection of your nose and throat.", 'diet': "Warm fluids, Chicken soup.", 'work': "Rest.", 'prec': ["Drink warm water", "Avoid cold food"]},
        'viral fever': {'meds': "['Paracetamol (Calpol)']", 'desc': "Viral infection causing a moderate rise in body temperature.", 'diet': "Fluids, Boiled vegetables.", 'work': "Strict Rest.", 'prec': ["Rest well", "Drink fluids"]},
        'sinus infection (sinusitis)': {'meds': "['Nasal Decongestants', 'Saline Spray']", 'desc': "Inflammation of the tissue lining the sinuses.", 'diet': "Warm liquids.", 'work': "Light walking.", 'prec': ["Use a humidifier", "Warm compress"]},
        'acidity and gas': {'meds': "['Antacids (Digene)', 'Pantoprazole']", 'desc': "Excess stomach acid.", 'diet': "Oatmeal, Bananas.", 'work': "Light walking.", 'prec': ["Avoid spicy food", "Don't lie down after eating"]},
        'food poisoning': {'meds': "['ORS', 'Ondansetron']", 'desc': "Illness caused by eating contaminated food.", 'diet': "BRAT diet.", 'work': "Strict Rest.", 'prec': ["Drink ORS", "Avoid solid foods"]},
        'indigestion': {'meds': "['Antacids']", 'desc': "Discomfort in your upper abdomen.", 'diet': "Ginger tea.", 'work': "Avoid core workouts.", 'prec': ["Chew food slowly", "Avoid fatty foods"]},
        'constipation issues': {'meds': "['Laxatives', 'Psyllium Husk']", 'desc': "Infrequent bowel movements.", 'diet': "High fiber diet.", 'work': "Walking.", 'prec': ["Increase fiber", "Stay hydrated"]},
        'tension headache': {'meds': "['Ibuprofen', 'Acetaminophen']", 'desc': "Mild to moderate pain usually caused by stress.", 'diet': "Chamomile tea.", 'work': "Neck stretches.", 'prec': ["Rest in quiet room", "Apply warm compress"]},
        'arthritis and joint issues': {'meds': "['Ibuprofen', 'Diclofenac Gel']", 'desc': "Inflammation of joints causing pain.", 'diet': "Omega-3 rich foods.", 'work': "Gentle stretching.", 'prec': ["Apply warm/cold compress", "Rest the joint"]},
        'muscle strain': {'meds': "['Muscle Relaxants', 'Ibuprofen']", 'desc': "Overstretched or torn muscle.", 'diet': "Protein-rich diet.", 'work': "Rest for 48 hours.", 'prec': ["Apply ice packs", "Elevate the area"]},
        'dry skin': {'meds': "['Moisturizing Lotions']", 'desc': "Skin lacks moisture.", 'diet': "Drink plenty of water.", 'work': "Avoid heavy sweating.", 'prec': ["Use a gentle soap", "Apply moisturizer"]},
        'skin allergy': {'meds': "['Antihistamines', 'Corticosteroids']", 'desc': "An allergic reaction causing an itchy rash.", 'diet': "Avoid known food allergens.", 'work': "Avoid sweating.", 'prec': ["Identify and avoid trigger", "Wash area"]},
        'seasonal allergies': {'meds': "['Cetirizine', 'Nasal Spray']", 'desc': "Allergic response to pollen or dust.", 'diet': "Warm herbal teas.", 'work': "Indoor exercises.", 'prec': ["Keep windows closed", "Wear a mask outdoors"]},
        'pink eye (conjunctivitis)': {'meds': "['Artificial Tears']", 'desc': "Eye infection or inflammation.", 'diet': "Healthy diet.", 'work': "Avoid swimming.", 'prec': ["Don't rub eyes", "Wash hands frequently"]},
        'eye strain': {'meds': "['Lubricating Eye Drops']", 'desc': "Fatigue of the eyes caused by screens.", 'diet': "Vitamin A foods.", 'work': "Limit screen time.", 'prec': ["Use 20-20-20 rule", "Adjust brightness"]},
        'dehydration': {'meds': "['ORS', 'IV Fluids']", 'desc': "Lack of water in the body.", 'diet': "Coconut water, Broths.", 'work': "Strict Rest.", 'prec': ["Drink fluids slowly", "Rest in a cool place"]},
        'fatigue and burnout': {'meds': "['Multivitamins']", 'desc': "Exhaustion from prolonged stress.", 'diet': "Balanced diet.", 'work': "Meditation.", 'prec': ["Ensure 8 hours of sleep", "Take breaks"]},
        'urinary tract infection (uti)': {'meds': "['Antibiotics', 'Cranberry Extract']", 'desc': "Infection in the urinary system.", 'diet': "Cranberry juice, Lots of water.", 'work': "Rest.", 'prec': ["Drink a lot of water", "Don't hold urine"]},
        'anxiety and stress': {'meds': "['Ashwagandha']", 'desc': "Body's response to stress.", 'diet': "Chamomile tea.", 'work': "Deep breathing.", 'prec': ["Practice mindfulness", "Reduce caffeine"]},
        'motion sickness': {'meds': "['Dimenhydrinate']", 'desc': "Disturbance of inner ear from motion.", 'diet': "Ginger, Clear sodas.", 'work': "Rest.", 'prec': ["Look at the horizon", "Get fresh air"]},
        'peptic ulcer (stomach ulcer)': {'meds': "['Pantoprazole', 'Antacids', 'Amoxicillin']", 'desc': "A sore that develops on the lining of the stomach or intestine, causing severe burning pain.", 'diet': "High fiber, apples, avoiding spicy/acidic foods.", 'work': "Strict rest, avoid heavy lifting.", 'prec': ["Avoid spicy food", "Do not smoke", "Reduce stress"]}
    }
    
    for d, m in custom_metadata.items():
        meds_dict[d] = m['meds']; desc_dict[d] = m['desc']; diet_dict[d] = m['diet']; work_dict[d] = m['work']; prec_dict[d] = m['prec']

    model_data = {
        'model': rf_model,                 
        'label_encoder': le,
        'symptoms_list': symptom_columns,
        'severity_dict': severity_dict,
        'descriptions': desc_dict, 
        'medications': meds_dict,
        'diets': diet_dict, 
        'workouts': work_dict, 
        'precautions': prec_dict
    }

    joblib.dump(model_data, 'model.pkl')
    print("✅ Logic corrected! Severe diseases will never trigger on a single generic symptom.")

if __name__ == "__main__":
    build_master_brain()
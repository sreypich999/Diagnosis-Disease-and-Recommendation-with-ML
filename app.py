import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title='Disease Prediction App', page_icon='🩺', layout='wide')

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f7f9fc;
        color: #333;
    }
    .sidebar .sidebar-content {
        background: #dee2e6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput, .stMultiSelect {
        border-radius: 0.25rem;
        border: 1px solid #ced4da;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333;
    }
    .section {
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .section:nth-child(odd) {
        background-color: #e9f7ef;
    }
    .section:nth-child(even) {
        background-color: #f2f3f4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# List of all symptoms in the dataset
all_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 
    'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
    'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 
    'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
    'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
    'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
    'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 
    'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 
    'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 
    'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine', 
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
    'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 
    'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 
    'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 
    'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 
    'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 
    'red_sore_around_nose', 'yellow_crust_ooze', 'fever'
]

# Load the description, precautions, diets, doctor, and workout data
description_df = pd.read_csv('description.csv')
precautions_df = pd.read_csv('precautions_df.csv')
diets_df = pd.read_csv('diets.csv')
doctor_df = pd.read_csv('doctor.csv')
workout_df = pd.read_csv('workout_df.csv')

def create_input_vector(selected_symptoms):
    input_vector = [0] * len(all_symptoms)
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            input_vector[index] = 1
    return [input_vector]

def predict_disease(selected_symptoms):
    input_vector = create_input_vector(selected_symptoms)
    probabilities = model.predict_proba(input_vector)[0]
    diseases = model.classes_
    disease_probabilities = list(zip(diseases, probabilities))
    sorted_probabilities = sorted(disease_probabilities, key=lambda x: x[1], reverse=True)
    return sorted_probabilities[:10]  # Top 10 diseases

st.title('🩺 Disease Prediction Based on Symptoms')

# Allow user to select symptoms
selected_symptoms = st.multiselect('Select Symptoms', all_symptoms)

# Check if at least one symptom is selected
if selected_symptoms:
    top_diseases = predict_disease(selected_symptoms)
    disease_names = [disease[0] for disease in top_diseases]
    disease_probabilities = [disease[1] for disease in top_diseases]
    
    # Display top diseases and their probabilities in a bar chart
    st.subheader('Top 10 Diseases')
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=disease_probabilities, y=disease_names, ax=ax, palette='viridis')
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Disease', fontsize=12)
    ax.set_title('Top 10 Diseases Based on Symptoms', fontsize=15)
    sns.despine()
    st.pyplot(fig)
    plt.close(fig)  # Clear the figure after plotting to free up memory

    # Display description, precautions, diets, and doctor for the top disease
    top_disease = disease_names[0]

    col1, col2 = st.columns(2)

    with col1:
        # Display description
        if not description_df[description_df['Disease'] == top_disease].empty:
            disease_description = description_df[description_df['Disease'] == top_disease]['Description'].values[0]
            st.markdown(f'<div class="section"><h3>📄 Description of {top_disease}</h3><p>{disease_description}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="section"><h3>📄 Description of {top_disease}</h3><p>No description available for {top_disease}.</p></div>', unsafe_allow_html=True)

        # Display precautions
        if not precautions_df[precautions_df['Disease'] == top_disease].empty:
            disease_precautions = precautions_df[precautions_df['Disease'] == top_disease].iloc[:, 1:].values.tolist()[0]
            precautions_html = ''.join([f"<li>{precaution}</li>" for precaution in disease_precautions])
            st.markdown(f'<div class="section"><h3>⚠️ Precautions for {top_disease}</h3><ul>{precautions_html}</ul></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="section"><h3>⚠️ Precautions for {top_disease}</h3><p>No precautions available for {top_disease}.</p></div>', unsafe_allow_html=True)

    with col2:
        # Display diets
        if not diets_df[diets_df['Disease'] == top_disease].empty:
            disease_diets = diets_df[diets_df['Disease'] == top_disease].iloc[:, 1:].values.tolist()[0]
            diets_html = ''.join([f"<li>{diet}</li>" for diet in disease_diets])
            st.markdown(f'<div class="section"><h3>🥗 Diet for {top_disease}</h3><ul>{diets_html}</ul></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="section"><h3>🥗 Diet for {top_disease}</h3><p>No diet information available for {top_disease}.</p></div>', unsafe_allow_html=True)

        # Display workout
        top_disease_workout = workout_df[workout_df['disease'] == top_disease]
        if top_disease_workout.empty:
            st.markdown(f'<div class="section"><h3>💪 Workout for {top_disease}</h3><p>No workout information available for this disease.</p></div>', unsafe_allow_html=True)
        else:
            workout_html = ''.join([f"<li>{workout}</li>" for workout in top_disease_workout['workout'].values])
            st.markdown(f'<div class="section"><h3>💪 Workout for {top_disease}</h3><ul>{workout_html}</ul></div>', unsafe_allow_html=True)

    # Display doctor information
    top_disease_doctor = doctor_df[doctor_df['Disease'] == top_disease]
    if top_disease_doctor.empty:
        st.markdown(f'<div class="section"><h3>🩺 Doctor for {top_disease}</h3><p>No doctor information available for this disease.</p></div>', unsafe_allow_html=True)
    else:
        doctor_info_html = ''
        for idx, row in top_disease_doctor.iterrows():
            doctor_info_html += f"<li><strong>Doctor Name:</strong> {row['Doctor_name']}<br><strong>Workplace:</strong> {row['workplace']}<br><strong>Contact:</strong> {row['contact']}</li>"
        st.markdown(f'<div class="section"><h3>🩺 Doctor for {top_disease}</h3><ul>{doctor_info_html}</ul></div>', unsafe_allow_html=True)
else:
    st.write('Please select at least one symptom to proceed.')

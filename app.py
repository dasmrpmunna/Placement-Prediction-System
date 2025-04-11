from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('new_model.pkl', 'rb'))

# Feature columns (must match the training order)
feature_names = [
    'CGPA',
    'Internships',
    'Projects',
    'Workshops/Certifications',
    'AptitudeTestScore',
    'SoftSkillsRating',
    'ExtracurricularActivities',
    'PlacementTraining',
    'SSC_Marks',
    'HSC_Marks'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form values
    input_data = [request.form.get(col) for col in feature_names]

    # Convert to correct types
    numeric_cols = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
                    'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks']
    
    for i, col in enumerate(feature_names):
        if col in numeric_cols:
            input_data[i] = float(input_data[i])
    
    # Create DataFrame for prediction
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Make prediction
    result = model.predict(input_df)[0]
    
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)



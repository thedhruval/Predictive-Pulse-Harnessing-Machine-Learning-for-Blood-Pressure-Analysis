from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained classifier
model = pickle.load(open('best_random_forest_model.pkl', 'rb'))

# Mapping of model output → human-readable stages
decoded_stage = {
    0: "NORMAL",
    1: "HYPERTENSION (Stage-1)",
    2: "HYPERTENSION (Stage-2)",
    3: "HYPERTENSIVE CRISIS"
}

# Mapping of form inputs → encoded labels your model expects
gender_map       = {"Female": 0, "Male": 1}
yesno_map        = {"No": 0, "Yes": 1}
patient_map      = {"Inpatient": 0, "Outpatient": 1}
severity_map     = {"Mild": 0, "Moderate": 1, "Severe": 2}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1) Read form values
    form = request.form
    gender         = gender_map[form['gender']]
    age            = float(form['age'])
    history        = yesno_map[form['history']]
    patient        = patient_map[form['patient']]
    take_med       = yesno_map[form['take_med']]
    severity       = severity_map[form['severity']]
    breath_short   = yesno_map[form['breath_short']]
    visual_change  = yesno_map[form['visual_change']]
    nose_bleed     = yesno_map[form['nose_bleed']]
    diet           = yesno_map[form['diet']]
    systolic       = float(form['systolic'])
    diastolic      = float(form['diastolic'])

    # 2) Build feature array in the same order you trained your model
    features = np.array([
        gender, age, history, patient, take_med,
        severity, breath_short, visual_change,
        nose_bleed, diet, systolic, diastolic
    ]).reshape(1, -1)

    # 3) Predict and decode
    pred_code = model.predict(features)[0]
    stage_text = decoded_stage.get(pred_code, "Unknown")

    # 4) Render result page
    return render_template(
        'prediction.html',
        prediction_text=f"Estimated Blood Pressure Stage: {stage_text}"
    )

if __name__ == '__main__':
    app.run(debug=True)

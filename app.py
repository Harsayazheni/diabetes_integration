import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the scaler and model
scaler = joblib.load('./model/scaler.pkl')
model = joblib.load('./model/diabetes_classifier.pkl')

@app.route('/')
def index():
    return render_template('form.html')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = [float(request.form.get(attr)) for attr in ['attr1', 'attr2', 'attr3', 'attr4', 'attr5', 'attr6', 'attr7', 'attr8']]
        
        # Standardize the input data
        data = [data]  # Convert to 2D array
        scaled_data = scaler.transform(data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        
        # Return prediction result
        return jsonify({'prediction': 'Diabetic' if prediction == 1 else 'Non-diabetic'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

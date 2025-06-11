from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('F:/all_projects/projects/House_price/Housing_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_order = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
                         'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
        features = [float(request.form[feat]) for feat in feature_order]
        prediction = model.predict([np.array(features)])
        price = round(prediction[0], 2)
        return render_template('index.html', prediction_text=f"🏠 Predicted House Price: ₹{price} Lakhs")
    except Exception as e:
        return render_template('index.html', prediction_text=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

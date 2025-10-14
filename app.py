from flask import Flask, request, render_template
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils import load_object # We need this for the one-time load

app = Flask(__name__)

# --- THE FIX: Load the models once at startup ---
# These variables will be loaded into memory when the app starts
# and will be available globally within the application context.

try:
    vectorizer_path = 'artifacts/vectorizer.pkl'
    model_path = 'artifacts/model.pkl'
    
    # Load the objects into global variables
    vectorizer = load_object(file_path=vectorizer_path)
    model = load_object(file_path=model_path)
    
except Exception as e:
    raise CustomException(e, sys)
# --- END OF FIX ---


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        review_text = request.form.get('review_text')
        
        # Now, instead of loading the files, we use the objects already in memory
        vectorized_features = vectorizer.transform([review_text])
        prediction = model.predict(vectorized_features)
        
        result = "Positive" if prediction[0] == 1 else "Negative"

        return render_template('home.html', prediction_result=result, review_text=review_text)

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
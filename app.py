from flask import Flask, request, render_template
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

# Initialize prediction pipeline once
predict_pipeline = PredictPipeline()


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict_datapoint():
    try:
        review_text = request.form.get("review_text")

        # Wrap input using CustomData abstraction
        data = CustomData(text=review_text)
        features = data.get_data_for_prediction()

        # Run prediction through pipeline
        prediction = predict_pipeline.predict(features)[0]

        return render_template(
            "home.html",
            prediction_result=prediction,
            review_text=review_text
        )

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

# Mess Food Sentiment Analysis

A production-ready web application to analyze the sentiment of college mess food reviews. This project comes with pre-trained models, allowing for immediate use. It classifies a user's text review as either "Positive" or "Negative" through a clean web interface built with Flask.

## Features

* **Pre-Trained & Ready to Use:** Clone the repository and run the web app instantly without needing to train the model first.
* **Sentiment Classification:** Accurately classifies food reviews into Positive and Negative categories.
* **Web Interface:** A simple and intuitive UI built with Flask for real-time predictions.
* **Modular Codebase:** The project is structured with a clear separation of concerns, following best practices for production-ready applications.
* **Complete Training Pipeline:** Includes optional scripts to re-train the model from scratch, including data ingestion, preprocessing (lemmatization, stop-word removal), and evaluation.
* **Optimized for Performance:** The Flask application pre-loads the model into memory for instantaneous predictions.

## 📈 Workflow

This project is divided into two primary workflows: training and prediction.

### Training Pipeline (`train_pipeline.py`)

This workflow processes the raw data and creates the final machine learning model.

1.  **Data Ingestion:** Reads the raw `Reviews.csv`, filters out neutral reviews, converts 5-star ratings into binary sentiment (Positive/Negative), and splits the data into `train.csv` and `test.csv`.
2.  **Data Transformation:** The raw text from the train and test sets is loaded.
3.  **Model Training:** A Scikit-learn `Pipeline` is used to:
    * Apply text preprocessing (lemmatization, stop-word removal, n-grams).
    * Vectorize the text data using `TfidfVectorizer`.
    * Train a `LogisticRegression` classifier on the vectorized data.
4.  **Artifacts Saved:** The fitted vectorizer and model are saved as `.pkl` files in the `artifacts/` folder.

### Prediction Pipeline (`app.py`)

This workflow is executed when a user submits a review through the Flask web interface.

1.  **App Startup:** The Flask application starts and pre-loads the `vectorizer.pkl` and `model.pkl` files into memory a single time.
2.  **User Input:** A user enters a review into the web form and clicks "Analyze Sentiment."
3.  **Request Handling:** The Flask backend receives the raw text.
4.  **Prediction:** The pre-loaded vectorizer transforms the text, and the pre-loaded model predicts the sentiment.
5.  **Response:** The predicted sentiment ("Positive" or "Negative") is sent back to the user and displayed on the web page.

## 🚀 Tech Stack

* **Backend:** Flask
* **Machine Learning:** Scikit-learn, NLTK, Pandas, NumPy
* **Language:** Python 3

## ⚙️ Quick Start: Running the Pre-Trained Application

This project includes pre-trained model files, so you can run the web application right away.

**1. Clone the repository:**
```bash
git clone [https://github.com/MrVRai/sentiment_analysis.git](https://github.com/MrVRai/sentiment_analysis.git)
cd sentiment_analysis
```

**2. Create a virtual environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
**3. Install the required packages:**

The `-e .` command installs your project in "editable" mode, which is crucial for the modular structure to work correctly.
```bash
pip install -r requirements.txt
```

**4. Run the Flask Web Application:**
```bash
python app.py
```
Now, open your web browser and navigate to http://127.0.0.1:5000 to use the application!

## 🛠️ Re-Training the Model (Optional)

If you wish to re-train the model yourself, you must first provide the dataset.

**1. Download the Dataset:**

This project uses the "Amazon Fine Food Reviews" dataset from Kaggle.

* Download the dataset from this link: [**Amazon Fine Food Reviews on Kaggle**](https://www.kaggle.com/snap/amazon-fine-food-reviews)
* From the downloaded archive, extract the `Reviews.csv` file.

**2. Place the Dataset:**

Place the `Reviews.csv` file inside the `artifacts/` folder. The folder structure should look like this:
sentiment_analysis/
├── artifacts/
│   └── Reviews.csv  <-- The file goes here
├── src/
└── ...


**3. Run the Training Pipeline:**

Once the dataset is in the correct location, run the training script from the **root directory** of the project.
```bash
python -m src.pipeline.train_pipeline
```

This process will create new, updated `vectorizer.pkl` and `model.pkl` files in your `artifacts/` folder based on the data you provided.

## 📁 Project Structure

├── app.py              # Main Flask application
├── requirements.txt    # Project dependencies
├── setup.py            # Setup script for installing the project as a package
├── templates/
│   └── home.html       # HTML template for the UI
├── artifacts/
│   ├── model.pkl       # Pre-trained model
│   └── vectorizer.pkl  # Pre-trained vectorizer
└── src/
├── components/     # Individual ML pipeline components
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   └── model_trainer.py
└── pipeline/
├── predict_pipeline.py # Prediction logic
└── train_pipeline.py   # Script to run the training pipeline


## 🙏 Acknowledgements

A special thanks to the creators of the Amazon Fine Food Reviews dataset and the powerful op
# End-to-End NLP Feedback Intelligence Platform

A **production-style Machine Learning system** for sentiment analysis of user reviews, covering the **complete ML lifecycle** — data ingestion, NLP preprocessing, model comparison, evaluation, artifact persistence, and deployment-ready inference.

This project demonstrates **end-to-end ML engineering ownership**, from raw data to real-time predictions, with a strong focus on **reproducibility, evaluation rigor, and system design**.

---

## 🔍 Project Overview

The system analyzes text reviews and classifies them as **Positive** or **Negative** sentiment.  
It is designed using **modular ML pipelines**, enabling clean separation between training and inference, fair model benchmarking, and scalable deployment.

The repository includes **pre-trained artifacts** for immediate use and supports **re-training from scratch** using the Amazon Fine Food Reviews dataset.

---

## ✨ Key Capabilities

- **End-to-End ML Pipeline:** Data ingestion → transformation → model training → evaluation → inference  
- **Model Comparison & Selection:** Benchmarks Logistic Regression and Linear SVM using F1-score, precision, and recall, and automatically selects the best-performing model  
- **NLP-Aware Feature Engineering:** TF-IDF with lemmatization, stop-word removal, and n-grams  
- **Reproducible Artifacts:** Persisted vectorizer and model for consistent inference  
- **Production-Ready Inference:** Decoupled prediction pipeline with artifact loading and batch-safe inference  
- **Web Integration:** Lightweight Flask interface for real-time predictions  

---

## 🧠 Machine Learning Workflow

### Training Pipeline (`train_pipeline.py`)

The training workflow is modular, reproducible, and evaluation-driven.

1. **Data Ingestion**
   - Reads raw `Reviews.csv`
   - Filters neutral (3-star) reviews
   - Converts ratings into binary sentiment labels
   - Performs stratified train–test split

2. **Data Transformation**
   - Applies NLP preprocessing (tokenization, lemmatization, stop-word removal)
   - Extracts TF-IDF features with uni-grams and bi-grams

3. **Model Training & Evaluation**
   - Trains **Logistic Regression** and **Linear SVM** on the same feature space
   - Evaluates models using **F1-score, precision, and recall**
   - Automatically selects and persists the best-performing model

4. **Artifacts Generated**
   - `model.pkl` – selected best model
   - `vectorizer.pkl` – fitted TF-IDF vectorizer
   - `model_metrics.csv` – model comparison table

---

### Inference Pipeline (`predict_pipeline.py`)

The inference workflow is fully decoupled from training.

1. Loads persisted model and vectorizer artifacts  
2. Applies the same feature transformation used during training  
3. Performs batch-safe prediction  
4. Maps numerical outputs to human-readable sentiment labels  

This design ensures **consistent, low-latency predictions** and safe reuse in APIs or agent-based systems.

---

## 🚀 Tech Stack

- **Language:** Python 3  
- **Machine Learning:** Scikit-learn, NLTK  
- **NLP:** TF-IDF, Lemmatization, n-grams  
- **Backend:** Flask  
- **Data Processing:** Pandas, NumPy  

---

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
```
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
```

## 🙏 Acknowledgements

A special thanks to the creators of the Amazon Fine Food Reviews dataset and the powerful op
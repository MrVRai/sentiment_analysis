# End-to-End NLP Feedback Intelligence Platform

A **production-style NLP system** for sentiment analysis that combines **classical machine learning and deep learning pipelines** with automated model comparison.

This project demonstrates **end-to-end ML engineering ownership**, from data ingestion to training, evaluation, model selection, and deployment-ready inference, with a strong focus on **reproducibility, evaluation rigor, and system design**.

---

## 🔍 Project Overview

The system analyzes text reviews and classifies them as **Positive** or **Negative** sentiment using both: 

- Classical ML models (TF-IDF + Logistic Regression / Linear SVM)
- Deep learning models (BiLSTM / BiGRU with self-attention)

A unified evaluation pipeline compares all models and automatically identifies the best-performing approach based on F1-score.

---

## ✨ Key Capabilities

- End-to-end ML pipelines (ingestion → transformation → training → evaluation)
- Dual modeling approach: Classical ML + Deep Learning
- Sequence modeling with embeddings, BiLSTM/BiGRU, and attention
- Automated multi-model benchmarking
- Cross-family model comparison (ML vs DL)
- Best-model selection for deployment
- Reproducible artifact generation
- Modular, production-style codebase
- Lightweight Flask interface for real-time predictions  

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Ingestion

- Loads Amazon Fine Food Reviews dataset  
- Removes neutral (3-star) reviews  
- Converts ratings to binary sentiment  
- Performs stratified train–test split  

---

### 2️⃣ Classical ML Pipeline

- TF-IDF vectorization (uni-grams + bi-grams)  
- Lemmatization and stop-word removal  
- Logistic Regression and Linear SVM training  
- Evaluation using F1, precision, and recall  

**Best Classical F1:** ~0.955  

---

### 3️⃣ Deep Learning Pipeline

- Tokenization and sequence padding  
- Embedding layers  
- Bidirectional LSTM and GRU  
- Self-attention mechanism  
- Validation tracking during training  

**Best Deep Learning F1:** ~0.98  

---

### 4️⃣ Model Selection

- Aggregates ML and DL metrics  
- Compares models across families  
- Automatically selects best-performing model  
- Saves comparison artifacts  

---

## 📊 Sample Results

| Model | Type | F1 Score |
|------|------|------|
BiGRU + Attention | Deep Learning | ~0.98  
BiLSTM + Attention | Deep Learning | ~0.98  
Linear SVM | Classical ML | ~0.955  
Logistic Regression | Classical ML | ~0.954  

---

## 🚀 Tech Stack

- Python  
- Scikit-learn  
- TensorFlow / Keras  
- NLTK  
- Pandas / NumPy  
- Flask (for inference UI)

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
```
sentiment_analysis/
├── artifacts/
│   └── Reviews.csv  <-- The file goes here
├── src/
└── ...
```

**3. Create a virtual environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**4. Install the required packages:**
```bash
pip install -r requirements.txt
```

**5. Run classical ML Pipeline:**
```bash
python -m src.pipeline.train_pipeline
```

**6. Run deep learning pipeline:**
```bash
python -m src.pipeline.dl_train_pipeline
```


## 📁 Project Structure
```
src/
├── components/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_trainer.py
│   ├── dl_data_transformation.py
│   ├── dl_model_trainer.py
│   └── model_selector.py
│
├── pipeline/
│   ├── train_pipeline.py
│   └── dl_train_pipeline.py
│
├── utils.py
├── logger.py
└── exception.py

```

## 🙏 Acknowledgements

- Dataset: Amazon Fine Food Reviews (Kaggle)
- Libraries: Scikit-learn, TensorFlow, NLTK

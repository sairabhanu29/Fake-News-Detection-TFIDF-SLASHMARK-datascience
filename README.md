# Fake-News-Detection-TFIDF-SLASHMARK-datascience
Fake news detection using TF-IDF and Liner model

# 📰 Fake News Detection using TF-IDF and Machine Learning

## 📌 Project Overview

Fake news has become a serious problem in the digital age, spreading misinformation and misleading people.
This project focuses on building a **machine learning–based text classification system** that can automatically identify whether a given news article is **Fake** or **Genuine (Real)**.

The model uses **Natural Language Processing (NLP)** techniques and **TF-IDF vectorization** along with a **linear machine learning classifier** to perform accurate classification.



## 🎯 Objectives

* To preprocess and clean news text data
* To convert text into numerical features using **TF-IDF**
* To train a **linear classifier** for fake news detection
* To evaluate the performance of the trained model
* To save the trained model for future predictions



## 🧠 Learning Outcomes

By completing this project, you will understand:

* Text preprocessing techniques in NLP
* TF-IDF vectorization
* Training and evaluating machine learning models
* Fake vs. real news classification
* Model persistence using `.sav` / `.pkl` files
  

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Development Environment:** Anaconda Jupyter Notebook
* **Libraries:**

  * pandas
  * numpy
  * scikit-learn
  * nltk (optional)



## 📂 Project Structure


Fake_News_Detection/
│
├── Fake_News_Detection_TFIDF.ipynb   # Main Jupyter Notebook
├── final_model.sav                  # Trained classification model
├── tfidf_vectorizer.sav             # Saved TF-IDF vectorizer
├── model.pkl                        # Alternative saved model
├── DataPrep.py                      # Data preprocessing script
├── FeatureSelection.py              # Feature selection logic
├── classifier.py                    # Model training script
├── prediction.py                    # Prediction script
├── train/                           # Training dataset
├── test/                            # Testing dataset
├── valid/                           # Validation dataset
├── liar_dataset/                    # Dataset source
├── README.md                        # Project documentation
└── LICENSE                          # License file


## 📊 Dataset Description

* Source: Kaggle / LIAR Dataset
* Contains news statements labeled as:

  * **0 → Fake**
  * **1 → Genuine**
* Main columns used:

  * `text` / `statement` → News content
  * `label` → Class label


## 🔍 Methodology

### 1️⃣ Data Loading

* Dataset loaded using **pandas**
* Unnecessary columns removed
* Missing values handled


### 2️⃣ Text Preprocessing

Performed the following steps:

* Converted text to lowercase
* Removed punctuation and numbers
* Removed stopwords
* Cleaned text stored in a new column (`clean_text`)


### 3️⃣ Feature Extraction (TF-IDF)

* Used **TfidfVectorizer**
* Converted cleaned text into numerical vectors
* Removed very frequent words using `max_df`


### 4️⃣ Model Training

* Used a **Linear Machine Learning Model**:

  * Logistic Regression / Linear SVM
* Split data into training and testing sets
* Model trained on TF-IDF features


### 5️⃣ Model Evaluation

Evaluated the model using:

* Accuracy score
* Classification report (Precision, Recall, F1-score)


### 6️⃣ Model Saving

* Trained model saved as:

  * `final_model.sav`
* TF-IDF vectorizer saved as:

  * `tfidf_vectorizer.sav`
* Enables reuse without retraining


## ✅ Results

* The model successfully classifies news articles as **Fake** or **Genuine**
* Achieved good accuracy on test data
* Demonstrates effectiveness of TF-IDF with linear models

## 🧪 Sample Output

Input News: "The government announced a new policy..."
Prediction: Genuine News


## ▶️ How to Run the Project

### Step 1: Clone the Repository


git clone https://github.com/your-username/Fake-News-Detection.git


### Step 2: Open Jupyter Notebook


jupyter notebook


### Step 3: Run the Notebook

Open `Fake_News_Detection_TFIDF.ipynb` and run all cells sequentially.


## 🔮 Future Enhancements

* Use deep learning models (LSTM, BERT)
* Add web interface using Flask or Streamlit
* Support multilingual news detection
* Improve accuracy using ensemble methods


## ⭐ Acknowledgements

* Kaggle Datasets
* Scikit-learn Documentation
* NLP Learning Resources

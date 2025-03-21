# Ai-Projects
# Movie Review Sentiment Analysis

This project focuses on performing sentiment analysis on the IMDB movie review dataset to classify reviews as positive or negative. It involves data preprocessing, building and comparing deep learning models (RNN and CNN), and evaluating their performance.

---

## **Project Overview**

The goal of this project is to analyze the sentiment of movie reviews from the IMDB dataset. The dataset contains 50,000 reviews labeled as either positive or negative. The project involves:

1. **Data Preprocessing**: Cleaning, tokenizing, and encoding the text data.
2. **Model Building**: Implementing and comparing two deep learning models:
   - **RNN (Recurrent Neural Network)**
   - **CNN (Convolutional Neural Network)**
3. **Model Evaluation**: Visualizing results using confusion matrices and accuracy/loss plots.
4. **Prediction**: Building a function to predict the sentiment of user-input reviews.

---

## **Installation and Setup**

### **Prerequisites**
- Python 3.x
- pip (Python package installer)

### **Steps to Run the Project**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/movie-review-sentiment-analysis.git
   cd movie-review-sentiment-analysis
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

3.**Download NLTK Data**:
   ```bash
   import nltk
   nltk.download('stopwords')

4.**Run the Jupyter Notebook or Python Script**:
Open the Jupyter Notebook (sentiment_analysis.ipynb) or Python script (sentiment_analysis.py).
Execute the cells or script to preprocess the data, train the models, and evaluate their performance.

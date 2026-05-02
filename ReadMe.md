# 📊 InsightLens — Product Review Analyzer

## 🔍 Overview

InsightLens is an end-to-end NLP-based application that analyzes product reviews to extract meaningful business insights.

Instead of just classifying reviews as positive or negative, this system identifies **why customers are dissatisfied** by discovering common complaint themes using topic modeling.

---

## 🎯 Key Features

- 📂 Upload product review datasets (CSV)
- 🧹 Text preprocessing and noise removal
- 😊 Sentiment analysis using VADER
- 🧠 Topic extraction using TF-IDF + NMF
- 🔄 Dynamic topic labeling (no hardcoded categories)
- 📊 Interactive dashboard built with Streamlit
- 💬 Displays real sample complaints for each issue

---

## 🧠 How It Works

1. **Data Input**
   - User uploads a CSV file containing product reviews

2. **Preprocessing**
   - Removes HTML tags, special characters, and noise
   - Normalizes text for better analysis

3. **Sentiment Analysis**
   - Uses VADER to classify reviews into:
     - Positive
     - Negative
     - Neutral

4. **Topic Modeling (Negative Reviews)**
   - Converts text into numerical features using TF-IDF
   - Applies NMF to extract hidden complaint topics

5. **Dynamic Topic Generation**
   - Extracts top keywords per topic
   - Generates human-readable issue labels automatically

6. **Visualization**
   - Displays:
     - Sentiment distribution
     - Top complaint categories
     - Key insights (% of issues)
     - Sample reviews

---

## 📊 Example Insights

- “37% of complaints are related to product/order issues”
- “27% of users report taste-related problems” and so on,..

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, NLTK, Scikit-learn  
- **NLP Techniques:** VADER, TF-IDF, NMF  
- **Frontend/UI:** Streamlit  

---

## 📁 Project Structure
product-analyzer-system/
│── app.py              # Streamlit application
│── requirements.txt    # Dependencies
│── README.md           # Project documentation


---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/bhuvanasankar24/InsightLens.git

cd InsightLens

pip install -r requirements.txt

streamlit run app.py
```
---

⚠️ Limitations
~ Topic modeling is unsupervised, so some noise in topics may occur
~ Works best with English text
~ Performance may slow with very large datasets

---

💡 Learnings
- Handling messy real-world text data
- Difference between rule-based and machine learning approaches
- Importance of interpretability in NLP systems
- Turning raw text into actionable business insights

---

Author

Bhuvaneshwari S

Final Year BSc Computer Science with AI,

LinkedIn - www.linkedin.com/in/bhuvana-sankar,

Mail - bhuvanaasankar241@gmail.com

---

This project is for educational purposes. You can use and modify it freely.